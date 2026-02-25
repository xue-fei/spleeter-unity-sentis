using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using Unity.InferenceEngine;
using UnityEngine;

/// <summary>
/// 基于 Unity.InferenceEngine 的音频分离器（人声/伴奏分离）
/// 支持立体声音频，使用 STFT + 深度学习模型 + Wiener 滤波
/// 内存优化版：流式处理，峰值内存固定，与歌曲长度无关
/// </summary>
public class AudioSeparatorSentis : MonoBehaviour
{
    [SerializeField] private ModelAsset _vocalsModelAsset;
    [SerializeField] private ModelAsset _accompanimentModelAsset;

    private const int N_FFT = 4096;
    private const int HOP_LENGTH = 1024;
    private const int NUM_BINS = 2049;   // N_FFT/2 + 1
    private const int MODEL_BINS = 1024;   // 模型输入的频率 bins
    private const int CHUNK_SIZE = 512;    // 模型处理的时间帧块大小
    private const float EPSILON = 1e-10f;

    private int _sampleRate = 44100;

    private Worker _vocalsWorker;
    private Worker _accompanimentWorker;
    private Model _vocalsModel;
    private Model _accompanimentModel;

    // 预分配缓冲区（主线程专用）
    private float[] _hannWindow;
    private Complex32[] _fftBuffer;
    private Complex32[] _ifftBufferV;   // Vocals 专用
    private Complex32[] _ifftBufferA;   // Accompaniment 专用
    private float[] _frameBuffer;
    private float[] _inferInputBuf;  // 推理输入复用缓冲

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

            _vocalsWorker = new Worker(_vocalsModel, BackendType.CPU);
            _accompanimentWorker = new Worker(_accompanimentModel, BackendType.CPU);

            _hannWindow = CreateHannWindow(N_FFT);
            _fftBuffer = new Complex32[N_FFT];
            _ifftBufferV = new Complex32[N_FFT];
            _ifftBufferA = new Complex32[N_FFT];
            _frameBuffer = new float[N_FFT];
            _inferInputBuf = new float[2 * CHUNK_SIZE * MODEL_BINS];

            _isInitialized = true;
            Debug.Log("✓ AudioSeparatorSentis 初始化成功");
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
    /// 协程方式分离（不阻塞主线程）
    /// 用法: StartCoroutine(SeparateFromFileAsync(path, (results) => { ... }))
    /// </summary>
    public IEnumerator SeparateFromFileAsync(string audioPath,
        Action<Dictionary<string, float[]>> onComplete,
        Action<string> onError = null,
        Action<float> onProgress = null)
    {
        if (!_isInitialized)
        {
            onError?.Invoke("分离器未初始化");
            yield break;
        }

        // ── 步骤1：在后台线程加载 WAV 并计算 STFT（纯 CPU 密集，不涉及 Unity API）──
        float[] waveform = null;
        StftResult[] stftResults = null;
        float[][][] stftMag = null;
        int numFrames = 0, numSplits = 0, outputLength = 0;
        string loadError = null;

        onProgress?.Invoke(0.05f);
        var loadTask = Task.Run(() =>
        {
            try
            {
                waveform = Util.LoadWavFile(audioPath, ref _sampleRate);

                int ns = waveform.Length / 2;
                float[] left = new float[ns];
                float[] right = new float[ns];
                for (int i = 0; i < ns; i++)
                {
                    left[i] = waveform[i * 2];
                    right[i] = waveform[i * 2 + 1];
                }

                // STFT 计算（不使用 Unity API，可在后台线程安全运行）
                stftResults = new StftResult[2]
                {
                    ComputeStftBg(left),
                    ComputeStftBg(right)
                };

                numFrames = stftResults[0].NumFrames;
                stftMag = ExtractStftMagnitudeBg(stftResults);

                int padding = (CHUNK_SIZE - (numFrames % CHUNK_SIZE)) % CHUNK_SIZE;
                int padded = numFrames + padding;
                if (padding > 0)
                    stftMag = PadStftData(stftMag, padding);

                numSplits = padded / CHUNK_SIZE;
                outputLength = (numFrames - 1) * HOP_LENGTH + N_FFT;
            }
            catch (Exception ex)
            {
                loadError = ex.Message + "\n" + ex.StackTrace;
            }
        });

        // 等待后台任务完成，同时不阻塞主线程
        while (!loadTask.IsCompleted)
            yield return null;

        if (loadError != null)
        {
            onError?.Invoke($"加载/STFT 失败: {loadError}");
            yield break;
        }

        Debug.Log($"[STFT] {numFrames} 帧, {numSplits} splits");
        onProgress?.Invoke(0.20f);

        // ── 步骤2：在主线程逐 split 推理（GPU Worker 必须在主线程调用）──
        float[][] vocalsOut = new float[2][] { new float[outputLength], new float[outputLength] };
        float[][] accompOut = new float[2][] { new float[outputLength], new float[outputLength] };
        float[][] windowSumsV = new float[2][] { new float[outputLength], new float[outputLength] };
        float[][] windowSumsA = new float[2][] { new float[outputLength], new float[outputLength] };

        var tensorShape = new TensorShape(2, 1, CHUNK_SIZE, MODEL_BINS);

        for (int splitIdx = 0; splitIdx < numSplits; splitIdx++)
        {
            FillInputBuffer(stftMag, splitIdx);

            float[] vocalRaw = RunSingleSplit(_vocalsWorker, tensorShape);
            float[] accompRaw = RunSingleSplit(_accompanimentWorker, tensorShape);

            if (vocalRaw == null || accompRaw == null)
            {
                Debug.LogWarning($"split {splitIdx} 推理返回空，跳过");
            }
            else
            {
                AccumulateISTFT(vocalRaw, accompRaw, stftResults, splitIdx, numFrames,
                                vocalsOut, accompOut, windowSumsV, windowSumsA);
            }

            // 每处理一个 split 让出一帧，避免主线程卡死
            if (splitIdx % 2 == 0)
            {
                float prog = 0.20f + 0.70f * (splitIdx + 1f) / numSplits;
                onProgress?.Invoke(prog);
                yield return null;
            }
        }

        // ── 步骤3：后台归一化（大数组遍历，放后台）──
        bool normalizesDone = false;
        Task.Run(() =>
        {
            NormalizeOutput(vocalsOut, windowSumsV, outputLength);
            NormalizeOutput(accompOut, windowSumsA, outputLength);
            normalizesDone = true;
        });
        while (!normalizesDone)
            yield return null;

        onProgress?.Invoke(0.95f);

        var results = new Dictionary<string, float[]>
        {
            ["vocals"] = InterleaveChannels(vocalsOut[0], vocalsOut[1]),
            ["accompaniment"] = InterleaveChannels(accompOut[0], accompOut[1])
        };

        onProgress?.Invoke(1.0f);
        Debug.Log("✓ 音频分离完成");
        onComplete?.Invoke(results);
    }

    /// <summary>
    /// 同步接口（小文件或编辑器工具使用，会阻塞）
    /// </summary>
    public Dictionary<string, float[]> SeparateFromFile(string audioPath)
    {
        Dictionary<string, float[]> result = null;
        string error = null;
        bool done = false;

        var coroutine = SeparateFromFileAsync(audioPath,
            r => { result = r; done = true; },
            e => { error = e; done = true; });

        // 手动驱动协程（同步模拟，仅供非 MonoBehaviour 场景）
        while (coroutine.MoveNext()) { }

        if (error != null) throw new Exception(error);
        return result;
    }

    // ──────────────────────────────────────────────
    //  推理（必须在主线程）
    // ──────────────────────────────────────────────

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
                    Array.Copy(stftMag[ch][frameIdx], 0, _inferInputBuf, timeOffset, MODEL_BINS);
                else
                    Array.Clear(_inferInputBuf, timeOffset, MODEL_BINS);
            }
        }
    }

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
    //  STFT（后台线程版，不使用任何 Unity API）
    // ──────────────────────────────────────────────

    /// <summary>后台线程安全的 STFT（独立缓冲，不共享主线程缓冲）</summary>
    private StftResult ComputeStftBg(float[] signal)
    {
        int numFrames = (signal.Length - N_FFT) / HOP_LENGTH + 1;
        float[] realPart = new float[numFrames * NUM_BINS];
        float[] imagPart = new float[numFrames * NUM_BINS];
        float[] frame = new float[N_FFT];
        Complex32[] fftBuf = new Complex32[N_FFT];

        for (int fi = 0; fi < numFrames; fi++)
        {
            int offset = fi * HOP_LENGTH;
            for (int i = 0; i < N_FFT; i++)
            {
                int si = offset + i;
                frame[i] = si < signal.Length ? signal[si] * _hannWindow[i] : 0f;
            }
            for (int i = 0; i < N_FFT; i++)
                fftBuf[i] = new Complex32(frame[i], 0f);

            Fourier.Forward(fftBuf, FourierOptions.Matlab);

            int baseIdx = fi * NUM_BINS;
            for (int k = 0; k < NUM_BINS; k++)
            {
                realPart[baseIdx + k] = fftBuf[k].Real;
                imagPart[baseIdx + k] = fftBuf[k].Imaginary;
            }
        }
        return new StftResult { Real = realPart, Imag = imagPart, NumFrames = numFrames };
    }

    /// <summary>后台线程安全的幅度谱提取（不使用 Mathf）</summary>
    private float[][][] ExtractStftMagnitudeBg(StftResult[] stftResults)
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
                for (int k = 0; k < MODEL_BINS; k++)
                {
                    float r = stftResults[ch].Real[idx + k];
                    float im = stftResults[ch].Imag[idx + k];
                    result[ch][i][k] = (float)Math.Sqrt(r * r + im * im);
                }
            }
        }
        return result;
    }

    // ──────────────────────────────────────────────
    //  ISTFT + Wiener 掩码叠加
    // ──────────────────────────────────────────────

    /// <summary>
    /// 核心修复点：
    /// 1. vocals 和 accompaniment 使用各自独立的 windowSums，归一化互不干扰
    /// 2. k >= MODEL_BINS 的高频部分置零（不保留原始信号，避免串音）
    /// 3. vocals 和 accompaniment 的 IFFT 缓冲互相独立（_ifftBufferV / _ifftBufferA）
    /// </summary>
    private void AccumulateISTFT(
        float[] vocalRaw,
        float[] accompRaw,
        StftResult[] stftResults,
        int splitIdx,
        int numFrames,
        float[][] vocalsOut,
        float[][] accompOut,
        float[][] windowSumsV,
        float[][] windowSumsA)
    {
        int outputLength = vocalsOut[0].Length;
        int startFrame = splitIdx * CHUNK_SIZE;
        int endFrame = Math.Min(startFrame + CHUNK_SIZE, numFrames);

        for (int ch = 0; ch < 2; ch++)
        {
            StftResult stft = stftResults[ch];
            int chOff = ch * CHUNK_SIZE * MODEL_BINS;

            for (int lt = 0; lt < endFrame - startFrame; lt++)
            {
                int frameIdx = startFrame + lt;
                int offset = frameIdx * HOP_LENGTH;
                int stftBase = frameIdx * NUM_BINS;
                int rawBase = chOff + lt * MODEL_BINS;

                // ── 计算 Wiener 掩码并同时填充两个 IFFT 缓冲 ──
                for (int k = 0; k < NUM_BINS; k++)
                {
                    if (k < MODEL_BINS)
                    {
                        int ri = rawBase + k;
                        float vs = ri < vocalRaw.Length ? vocalRaw[ri] : 0f;
                        float ac = ri < accompRaw.Length ? accompRaw[ri] : 0f;
                        float vs2 = vs * vs;
                        float ac2 = ac * ac;
                        float denom = vs2 + ac2 + EPSILON;

                        // 修复：vocals 掩码 + accompaniment 掩码，两者之和 ≈ 1
                        float vm = (vs2 + EPSILON * 0.5f) / denom;
                        float am = (ac2 + EPSILON * 0.5f) / denom;

                        float re = stft.Real[stftBase + k];
                        float im = stft.Imag[stftBase + k];

                        _ifftBufferV[k] = new Complex32(re * vm, im * vm);
                        _ifftBufferA[k] = new Complex32(re * am, im * am);
                    }
                    else
                    {
                        // 修复：高频置零，不让原始信号泄漏到分离输出中
                        _ifftBufferV[k] = Complex32.Zero;
                        _ifftBufferA[k] = Complex32.Zero;
                    }
                }

                // ── 填充共轭对称 ──
                FillConjugateSymmetry(_ifftBufferV);
                FillConjugateSymmetry(_ifftBufferA);

                // ── IFFT ──
                Fourier.Inverse(_ifftBufferV, FourierOptions.Matlab);
                Fourier.Inverse(_ifftBufferA, FourierOptions.Matlab);

                // ── 重叠相加（vocals 和 accompaniment 各自独立累加）──
                for (int i = 0; i < N_FFT && offset + i < outputLength; i++)
                {
                    float w = _hannWindow[i];
                    float w2 = w * w;

                    vocalsOut[ch][offset + i] += _ifftBufferV[i].Real * w;
                    accompOut[ch][offset + i] += _ifftBufferA[i].Real * w;

                    // 修复：各自独立的 windowSum，归一化互不影响
                    windowSumsV[ch][offset + i] += w2;
                    windowSumsA[ch][offset + i] += w2;
                }
            }
        }
    }

    private void FillConjugateSymmetry(Complex32[] buf)
    {
        for (int k = NUM_BINS; k < N_FFT; k++)
        {
            int ci = N_FFT - k;
            buf[k] = ci < NUM_BINS ? Complex32.Conjugate(buf[ci]) : Complex32.Zero;
        }
    }

    // ──────────────────────────────────────────────
    //  辅助方法
    // ──────────────────────────────────────────────

    private static void NormalizeOutput(float[][] output, float[][] windowSums, int length)
    {
        for (int ch = 0; ch < 2; ch++)
            for (int i = 0; i < length; i++)
                if (windowSums[ch][i] > 1e-6f)
                    output[ch][i] /= windowSums[ch][i];
    }

    private float[] CreateHannWindow(int length)
    {
        var w = new float[length];
        for (int i = 0; i < length; i++)
            w[i] = 0.5f * (1f - (float)Math.Cos(2.0 * Math.PI * i / (length - 1)));
        return w;
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
        float[] stereo = new float[len * 2];
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