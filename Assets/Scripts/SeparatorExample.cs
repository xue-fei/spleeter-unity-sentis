using UnityEngine;
using System.Collections;

public class SeparatorExample : MonoBehaviour
{
    private AudioSeparatorSentis _separator;
    private bool _isProcessing = false;

    void Start()
    {
        _separator = GetComponent<AudioSeparatorSentis>();
        if (_separator == null)
        {
            Debug.LogError("请在同一 GameObject 上挂载 AudioSeparatorSentis 组件");
            return;
        }
        StartCoroutine(PerformSeparation());
    }

    private IEnumerator PerformSeparation()
    {
        if (_isProcessing)
        {
            Debug.LogWarning("⚠ 正在处理中，请等待...");
            yield break;
        }
        _isProcessing = true;

        string audioPath = Application.dataPath + "/qi-feng-le-zh.wav";
        string outputDir = Application.dataPath + "/SeparatedAudio/";

        Debug.Log($"=== 开始音频分离 ===");
        Debug.Log($"输入: {audioPath}");

        if (!System.IO.File.Exists(audioPath))
        {
            Debug.LogError($"❌ 音频文件不存在: {audioPath}");
            _isProcessing = false;
            yield break;
        }

        // 使用协程接口，不阻塞主线程
        yield return StartCoroutine(_separator.SeparateFromFileAsync(
            audioPath,
            onComplete: (sources) =>
            {
                Debug.Log($"✓ 分离完成，开始保存...");
                Util.SaveToFile(sources, outputDir, 44100);
                Debug.Log($"✓ 文件已保存至: {outputDir}");
                _isProcessing = false;
            },
            onError: (err) =>
            {
                Debug.LogError($"❌ 分离失败: {err}");
                _isProcessing = false;
            },
            onProgress: (p) =>
            {
                Debug.Log($"  进度: {p * 100f:F0}%");
            }
        ));
    }
}