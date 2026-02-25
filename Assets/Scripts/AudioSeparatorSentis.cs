using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;
using System;
using System.Collections.Generic;
using Unity.InferenceEngine;
using UnityEngine;

/// <summary>
/// 基于 Unity.InferenceEngine 的音频分离器（人声/伴奏分离）
/// 支持立体声音频，使用 STFT + 深度学习模型 + Wiener 滤波
/// 内存优化版：流式处理，峰值内存固定，与歌曲长度无关
/// </summary>
public class AudioSeparatorSentis : MonoBehaviour
{
    // 模型配置
    [SerializeField] private ModelAsset _vocalsModelAsset;
    [SerializeField] private ModelAsset _accompanimentModelAsset;

    // STFT 参数
    private const int N_FFT = 4096;
    private const int HOP_LENGTH = 1024;
    private const int NUM_BINS = 2049;   // N_FFT/2 + 1
    private const int MODEL_BINS = 1024;   // 模型输入的频率bins
    private const int CHUNK_SIZE = 512;    // 模型处理的帧块大小（时间轴）
    private const float EPSILON = 1e-10f;

    private int _sampleRate = 44100;

    // InferenceEngine 资源
    private Worker _vocalsWorker;
    private Worker _accompanimentWorker;
    private Model _vocalsModel;
    private Model _accompanimentModel;

    // 预分配缓冲区
    private float[] _hannWindow;
    private Complex32[] _fftBuffer;
    private Complex32[] _ifftBuffer;
    private float[] _frameBuffer;

    // 推理输入缓冲区（复用，避免每次 new）
    // 形状对应 (2, 1, CHUNK_SIZE, MODEL_BINS)
    private float[] _inferInputBuf;

    private bool _isInitialized = false;

    // ──────────────────────────────────────────────
    //  生命周期
    // ──────────────────────────────────────────────

    void Awake()
    {
        try
        {
            _vocalsModel = ModelLoader.Load(_vocalsModelAsset);
            _accompanimentModel = ModelLoader.Load(_accompanimentModelAsset);

            _vocalsWorker = new Worker(_vocalsModel, BackendType.GPUCompute);
            _accompanimentWorker = new Worker(_accompanimentModel, BackendType.GPUCompute);

            _hannWindow = CreateHannWindow(N_FFT);
            _fftBuffer = new Complex32[N_FFT];
            _ifftBuffer = new Complex32[N_FFT];
            _frameBuffer = new float[N_FFT];
            _inferInputBuf = new float[2 * CHUNK_SIZE * MODEL_BINS]; // 复用缓冲

            _isInitialized = true;
            Debug.Log("✓ AudioSeparatorSentis 初始化成功 (InferenceEngine GPU)");
        }
        catch (Exception ex)
        {
            Debug.LogError($"✗ 初始化失败: {ex.Message}\n{ex.StackTrace}");
            throw;
        }
    }

    void OnDestroy()
    {
        _vocalsWorker?.Dispose();
        _accompanimentWorker?.Dispose();
    }

    // ──────────────────────────────────────────────
    //  公开接口
    // ──────────────────────────────────────────────

    /// <summary>
    /// 从 WAV 文件加载并分离
    /// </summary>
    public Dictionary<string, float[]> SeparateFromFile(string audioPath)
    {
        try
        {
            float[] waveform = Util.LoadWavFile(audioPath, ref _sampleRate);
            return Separate(waveform);
        }
        catch (Exception ex)
        {
            Debug.LogError($"文件分离失败: {ex.Message}");
            throw;
        }
    }

    /// <summary>
    /// 分离立体声音频波形（44.1 kHz 16-bit PCM）
    /// </summary>
    /// <param name="waveform">交错立体声: [L0, R0, L1, R1, ...]</param>
    /// <returns>{"vocals", "accompaniment"} 各为交错立体声</returns>
    public Dictionary<string, float[]> Separate(float[] waveform)
    {
        if (!_isInitialized)
            throw new InvalidOperationException("分离器未初始化，请检查模型资源");

        var sw = System.Diagnostics.Stopwatch.StartNew();

        // ── 1. 解交错为左右声道 ──
        int numSamples = waveform.Length / 2;
        float[] left = new float[numSamples];
        float[] right = new float[numSamples];
        for (int i = 0; i < numSamples; i++)
        {
            left[i] = waveform[i * 2];
            right[i] = waveform[i * 2 + 1];
        }
        Debug.Log($"[1] 立体声解交错: {numSamples} 样本/通道");

        // ── 2. STFT ──
        var stftL = ComputeStft(left);
        var stftR = ComputeStft(right);
        var stftResults = new StftResult[2] { stftL, stftR };
        int numFrames = stftL.NumFrames;
        Debug.Log($"[2] STFT 完成: {numFrames} 帧");

        // ── 3. 提取幅度谱 [2][numFrames][MODEL_BINS] ──
        float[][][] stftMag = ExtractStftMagnitude(stftResults);

        // ── 4. 填充到 CHUNK_SIZE 整数倍 ──
        int padding = (CHUNK_SIZE - (numFrames % CHUNK_SIZE)) % CHUNK_SIZE;
        int paddedFrames = numFrames + padding;
        if (padding > 0)
            stftMag = PadStftData(stftMag, padding);
        int numSplits = paddedFrames / CHUNK_SIZE;
        Debug.Log($"[3] Padding: +{padding} 帧 → {paddedFrames} 帧, {numSplits} splits");

        // ── 5. 预分配输出缓冲区（固定大小，与歌曲长度成正比但只分配一次）──
        int outputLength = (numFrames - 1) * HOP_LENGTH + N_FFT;
        float[][] vocalsOut = new float[2][] { new float[outputLength], new float[outputLength] };
        float[][] accompOut = new float[2][] { new float[outputLength], new float[outputLength] };
        float[][] windowSums = new float[2][] { new float[outputLength], new float[outputLength] };

        // ── 6. 流式推理：逐 split 推理 → 立即 ISTFT 累加 ──
        var tensorShape = new TensorShape(2, 1, CHUNK_SIZE, MODEL_BINS);

        for (int splitIdx = 0; splitIdx < numSplits; splitIdx++)
        {
            // 填充推理输入（复用 _inferInputBuf）
            FillInputBuffer(stftMag, splitIdx);

            // 推理（两个模型共用同一输入缓冲，顺序执行）
            float[] vocalRaw = RunSingleSplit(_vocalsWorker, tensorShape);
            float[] accompRaw = RunSingleSplit(_accompanimentWorker, tensorShape);

            if (vocalRaw == null || accompRaw == null)
            {
                Debug.LogWarning($"split {splitIdx} 推理返回空结果，跳过");
                continue;
            }

            // 立即计算 Wiener 掩码 + ISTFT + 叠加到输出缓冲
            AccumulateISTFT(vocalRaw, accompRaw, stftResults, splitIdx,
                            numFrames, vocalsOut, accompOut, windowSums);

            if (splitIdx % 10 == 0)
                Debug.Log($"  推理进度: {splitIdx + 1}/{numSplits}");
        }

        // ── 7. 重叠相加归一化 ──
        for (int ch = 0; ch < 2; ch++)
        {
            for (int i = 0; i < outputLength; i++)
            {
                float w = windowSums[ch][i];
                if (w > 1e-6f)
                {
                    vocalsOut[ch][i] /= w;
                    accompOut[ch][i] /= w;
                }
            }
        }
        Debug.Log("[7] 归一化完成");

        // ── 8. 交错输出为立体声 ──
        var results = new Dictionary<string, float[]>
        {
            ["vocals"] = InterleaveChannels(vocalsOut[0], vocalsOut[1]),
            ["accompaniment"] = InterleaveChannels(accompOut[0], accompOut[1])
        };

        sw.Stop();
        float dur = numSamples / (float)_sampleRate;
        Debug.Log($"✓ 分离完成 | 耗时: {sw.ElapsedMilliseconds} ms | RTF: {sw.ElapsedMilliseconds / 1000f / dur:F3}");
        return results;
    }

    // ──────────────────────────────────────────────
    //  推理核心
    // ──────────────────────────────────────────────

    /// <summary>
    /// 将第 splitIdx 个 split 的幅度谱写入 _inferInputBuf
    /// 布局: [ch, time, freq]，对应张量形状 (2, 1, CHUNK_SIZE, MODEL_BINS)
    /// </summary>
    private void FillInputBuffer(float[][][] stftMag, int splitIdx)
    {
        for (int ch = 0; ch < 2; ch++)
        {
            int chOffset = ch * CHUNK_SIZE * MODEL_BINS;
            for (int time = 0; time < CHUNK_SIZE; time++)
            {
                int frameIdx = splitIdx * CHUNK_SIZE + time;
                int timeOffset = chOffset + time * MODEL_BINS;
                if (frameIdx < stftMag[ch].Length)
                {
                    float[] src = stftMag[ch][frameIdx];
                    Array.Copy(src, 0, _inferInputBuf, timeOffset, MODEL_BINS);
                }
                else
                {
                    Array.Clear(_inferInputBuf, timeOffset, MODEL_BINS);
                }
            }
        }
    }

    /// <summary>
    /// 用当前 _inferInputBuf 内容执行单次推理，立即释放 tensor，返回输出数组
    /// </summary>
    private float[] RunSingleSplit(Worker worker, TensorShape shape)
    {
        Tensor<float> inputTensor = null;
        Tensor<float> outputTensor = null;
        try
        {
            inputTensor = new Tensor<float>(shape, _inferInputBuf);
            worker.SetInput("x", inputTensor);
            worker.Schedule();

            outputTensor = worker.PeekOutput("y") as Tensor<float>;
            return outputTensor?.DownloadToArray();
        }
        catch (Exception ex)
        {
            Debug.LogError($"推理异常: {ex.Message}");
            return null;
        }
        finally
        {
            inputTensor?.Dispose();
            outputTensor?.Dispose();
        }
    }

    // ──────────────────────────────────────────────
    //  STFT / ISTFT
    // ──────────────────────────────────────────────

    private StftResult ComputeStft(float[] signal)
    {
        int numFrames = (signal.Length - N_FFT) / HOP_LENGTH + 1;
        float[] realPart = new float[numFrames * NUM_BINS];
        float[] imagPart = new float[numFrames * NUM_BINS];

        for (int frameIdx = 0; frameIdx < numFrames; frameIdx++)
        {
            int offset = frameIdx * HOP_LENGTH;

            for (int i = 0; i < N_FFT; i++)
            {
                int sIdx = offset + i;
                _frameBuffer[i] = sIdx < signal.Length ? signal[sIdx] * _hannWindow[i] : 0f;
            }

            for (int i = 0; i < N_FFT; i++)
                _fftBuffer[i] = new Complex32(_frameBuffer[i], 0f);

            Fourier.Forward(_fftBuffer, FourierOptions.Matlab);

            int baseIdx = frameIdx * NUM_BINS;
            for (int k = 0; k < NUM_BINS; k++)
            {
                realPart[baseIdx + k] = _fftBuffer[k].Real;
                imagPart[baseIdx + k] = _fftBuffer[k].Imaginary;
            }
        }

        return new StftResult { Real = realPart, Imag = imagPart, NumFrames = numFrames };
    }

    /// <summary>
    /// 对单个 split 的推理输出计算 Wiener 掩码，并将 ISTFT 结果叠加到输出缓冲区
    /// 不分配任何持久中间数组，内存使用量为 O(1)
    /// </summary>
    private void AccumulateISTFT(
        float[] vocalRaw,
        float[] accompRaw,
        StftResult[] stftResults,
        int splitIdx,
        int numFrames,
        float[][] vocalsOut,
        float[][] accompOut,
        float[][] windowSums)
    {
        int outputLength = vocalsOut[0].Length;
        int startFrame = splitIdx * CHUNK_SIZE;
        int endFrame = Mathf.Min(startFrame + CHUNK_SIZE, numFrames);

        for (int ch = 0; ch < 2; ch++)
        {
            StftResult stft = stftResults[ch];
            int chOff = ch * CHUNK_SIZE * MODEL_BINS;

            for (int localTime = 0; localTime < endFrame - startFrame; localTime++)
            {
                int frameIdx = startFrame + localTime;
                int offset = frameIdx * HOP_LENGTH;
                int stftBase = frameIdx * NUM_BINS;
                int rawBase = chOff + localTime * MODEL_BINS;

                // ── Vocals IFFT ──
                for (int k = 0; k < NUM_BINS; k++)
                {
                    if (k < MODEL_BINS)
                    {
                        int ri = rawBase + k;
                        float vs = ri < vocalRaw.Length ? vocalRaw[ri] : 0f;
                        float as_ = ri < accompRaw.Length ? accompRaw[ri] : 0f;
                        float vs2 = vs * vs, as2 = as_ * as_;
                        float vm = (vs2 + EPSILON * 0.5f) / (vs2 + as2 + EPSILON);
                        _ifftBuffer[k] = new Complex32(
                            stft.Real[stftBase + k] * vm,
                            stft.Imag[stftBase + k] * vm);
                    }
                    else
                    {
                        _ifftBuffer[k] = new Complex32(
                            stft.Real[stftBase + k],
                            stft.Imag[stftBase + k]);
                    }
                }
                FillConjugateSymmetry();
                Fourier.Inverse(_ifftBuffer, FourierOptions.Matlab);
                for (int i = 0; i < N_FFT && offset + i < outputLength; i++)
                {
                    float w = _hannWindow[i];
                    vocalsOut[ch][offset + i] += _ifftBuffer[i].Real * w;
                    windowSums[ch][offset + i] += w * w;   // windowSum 只累加一次
                }

                // ── Accompaniment IFFT ──
                for (int k = 0; k < NUM_BINS; k++)
                {
                    if (k < MODEL_BINS)
                    {
                        int ri = rawBase + k;
                        float vs = ri < vocalRaw.Length ? vocalRaw[ri] : 0f;
                        float as_ = ri < accompRaw.Length ? accompRaw[ri] : 0f;
                        float vs2 = vs * vs, as2 = as_ * as_;
                        float am = (as2 + EPSILON * 0.5f) / (vs2 + as2 + EPSILON);
                        _ifftBuffer[k] = new Complex32(
                            stft.Real[stftBase + k] * am,
                            stft.Imag[stftBase + k] * am);
                    }
                    else
                    {
                        _ifftBuffer[k] = new Complex32(
                            stft.Real[stftBase + k],
                            stft.Imag[stftBase + k]);
                    }
                }
                FillConjugateSymmetry();
                Fourier.Inverse(_ifftBuffer, FourierOptions.Matlab);
                for (int i = 0; i < N_FFT && offset + i < outputLength; i++)
                    accompOut[ch][offset + i] += _ifftBuffer[i].Real * _hannWindow[i];
                // accomp 与 vocals 共用 windowSums，不重复累加
            }
        }
    }

    private void FillConjugateSymmetry()
    {
        for (int k = NUM_BINS; k < N_FFT; k++)
        {
            int conjIdx = N_FFT - k;
            _ifftBuffer[k] = conjIdx < NUM_BINS
                ? Complex32.Conjugate(_ifftBuffer[conjIdx])
                : Complex32.Zero;
        }
    }

    // ──────────────────────────────────────────────
    //  辅助方法
    // ──────────────────────────────────────────────

    private float[] CreateHannWindow(int length)
    {
        var w = new float[length];
        for (int i = 0; i < length; i++)
            w[i] = 0.5f * (1f - Mathf.Cos(2f * Mathf.PI * i / (length - 1)));
        return w;
    }

    private float[][][] ExtractStftMagnitude(StftResult[] stftResults)
    {
        var result = new float[2][][];
        for (int ch = 0; ch < 2; ch++)
        {
            int nf = stftResults[ch].NumFrames;
            result[ch] = new float[nf][];
            for (int i = 0; i < nf; i++)
            {
                result[ch][i] = new float[MODEL_BINS];
                int idx = i * NUM_BINS;
                for (int k = 0; k < MODEL_BINS && k < NUM_BINS; k++)
                {
                    float r = stftResults[ch].Real[idx + k];
                    float im = stftResults[ch].Imag[idx + k];
                    result[ch][i][k] = Mathf.Sqrt(r * r + im * im);
                }
            }
        }
        return result;
    }

    private float[][][] PadStftData(float[][][] data, int padding)
    {
        int numFrames = data[0].Length;
        int newFrames = numFrames + padding;
        var padded = new float[2][][];
        for (int ch = 0; ch < 2; ch++)
        {
            padded[ch] = new float[newFrames][];
            Array.Copy(data[ch], 0, padded[ch], 0, numFrames);
            for (int i = numFrames; i < newFrames; i++)
                padded[ch][i] = new float[MODEL_BINS];
        }
        return padded;
    }

    private float[] InterleaveChannels(float[] left, float[] right)
    {
        int len = Math.Max(left.Length, right.Length);
        var stereo = new float[len * 2];
        for (int i = 0; i < len; i++)
        {
            stereo[i * 2] = i < left.Length ? left[i] : 0f;
            stereo[i * 2 + 1] = i < right.Length ? right[i] : 0f;
        }
        return stereo;
    }

    // ──────────────────────────────────────────────
    //  数据结构
    // ──────────────────────────────────────────────

    public class StftResult
    {
        public float[] Real;
        public float[] Imag;
        public int NumFrames;
    }
}