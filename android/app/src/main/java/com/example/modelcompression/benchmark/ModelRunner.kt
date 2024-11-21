package com.example.modelcompression.benchmark

import android.content.Context
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.nnapi.NnApiDelegate
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

class ModelRunner(private val context: Context) {
    private var tflite: Interpreter? = null
    private var gpuDelegate: GpuDelegate? = null
    private var nnapiDelegate: NnApiDelegate? = null
    
    companion object {
        private const val TAG = "ModelRunner"
        private const val MODEL_FILENAME = "compressed_model.tflite"
    }
    
    fun initializeModel(
        useGpu: Boolean = true,
        useNnapi: Boolean = true,
        socType: SocType = SocType.DEFAULT
    ) {
        try {
            val options = Interpreter.Options()
            
            // Try GPU delegation if requested
            if (useGpu && CompatibilityList().isDelegateSupportedOnThisDevice) {
                gpuDelegate = GpuDelegate()
                options.addDelegate(gpuDelegate)
                Log.i(TAG, "GPU delegation enabled")
            }
            
            // Try NNAPI delegation with SoC-specific configurations
            if (useNnapi) {
                try {
                    val nnApiOptions = NnApiDelegate.Options().apply {
                        // Base configuration
                        allowFp16 = true
                        useNnapiCpu = true
                        
                        // SoC-specific configurations
                        when (socType) {
                            SocType.QUALCOMM -> {
                                acceleratorName = "qti-dsp"  // Qualcomm DSP
                                // Additional Qualcomm-specific settings
                                setExecutionPreference(NnApiDelegate.PREFERENCE_PERFORMANCE)
                                setModelOptimizationPreference(NnApiDelegate.OPTIMIZATION_PERFORMANCE)
                            }
                            SocType.MEDIATEK -> {
                                acceleratorName = "mtk-npu"  // MediaTek NPU
                                // Additional MediaTek-specific settings
                                setExecutionPreference(NnApiDelegate.PREFERENCE_PERFORMANCE)
                            }
                            SocType.SAMSUNG -> {
                                acceleratorName = "samsung-npu"  // Samsung NPU
                                // Additional Samsung-specific settings
                                setExecutionPreference(NnApiDelegate.PREFERENCE_PERFORMANCE)
                            }
                            SocType.GOOGLE -> {
                                acceleratorName = "google-edgetpu"  // Google Edge TPU
                                // Additional Google-specific settings
                                setExecutionPreference(NnApiDelegate.PREFERENCE_PERFORMANCE)
                            }
                            SocType.DEFAULT -> {
                                // Default settings, let NNAPI choose the best accelerator
                                setExecutionPreference(NnApiDelegate.PREFERENCE_PERFORMANCE)
                            }
                        }
                    }
                    
                    nnapiDelegate = NnApiDelegate(nnApiOptions)
                    options.addDelegate(nnapiDelegate)
                    Log.i(TAG, "NNAPI delegation enabled for ${socType.name}")
                } catch (e: Exception) {
                    Log.w(TAG, "NNAPI delegation failed: ${e.message}")
                }
            }
            
            // Load model
            val modelFile = File(context.getExternalFilesDir(null), MODEL_FILENAME)
            tflite = Interpreter(modelFile, options)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error initializing model: ${e.message}")
            throw e
        }

    fun runInference(input: ByteBuffer): InferenceResult {
        val startTime = SystemClock.elapsedRealtimeNanos()
        
        // Prepare output buffer
        val outputBuffer = ByteBuffer.allocateDirect(1000 * 4).apply {
            order(ByteOrder.nativeOrder())
        }
        
        // Run inference
        tflite?.run(input, outputBuffer)
        
        val endTime = SystemClock.elapsedRealtimeNanos()
        return InferenceResult(endTime - startTime)
    }
    
    fun close() {
        tflite?.close()
        gpuDelegate?.close()
        nnapiDelegate?.close()
    }
}
