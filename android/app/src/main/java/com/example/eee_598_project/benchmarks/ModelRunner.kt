package com.example.eee_598_project.benchmarks

import android.content.Context
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.MappedByteBuffer

class ModelRunner(private val context: Context) {
    private var tflite: Interpreter? = null
    private lateinit var outputTensorBuffer: TensorBuffer

    companion object {
        private const val TAG = "ModelRunner"
    }

    fun initializeModel(modelFileName: String) {
        try {
            // Load the TFLite model
            val modelFile: MappedByteBuffer = FileUtil.loadMappedFile(context, modelFileName)
            tflite = Interpreter(modelFile)

            Log.i(TAG, "Model $modelFileName loaded successfully")
        } catch (e: IOException) {
            Log.e(TAG, "Error loading model: ${e.message}")
            throw e
        }
    }

    fun runInference(inputImage: TensorImage): InferenceResult {
        if (tflite == null) throw IllegalStateException("Model not initialized")


        // Retrieve the expected input tensor shape
        val inputTensor = tflite?.getInputTensor(0)
            ?: throw IllegalArgumentException("Failed to retrieve input tensor")
        val inputShape = inputTensor.shape()
        val expectedBufferSize = inputTensor.numBytes()

        // Ensure image dimensions match the model's expected dimensions
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(inputShape[1], inputShape[2], ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f)) // Normalize pixel values between 0 and 1
            .build()

        val processedImage = imageProcessor.process(inputImage)

        // Validate the buffer size
        if (processedImage.buffer.remaining() != expectedBufferSize) {
            throw IllegalArgumentException(
                "Buffer size mismatch: Expected $expectedBufferSize bytes but got ${processedImage.buffer.remaining()} bytes."
            )
        }

        // Allocate buffer for output
        val outputTensor = tflite?.getOutputTensor(0)
            ?: throw IllegalArgumentException("Failed to retrieve output tensor")
        outputTensorBuffer = TensorBuffer.createFixedSize(outputTensor.shape(), outputTensor.dataType())

        // Run inference
        val startTime = SystemClock.elapsedRealtimeNanos()
        tflite?.run(processedImage.buffer, outputTensorBuffer.buffer.rewind())
        val endTime = SystemClock.elapsedRealtimeNanos()

        return InferenceResult(
            inferenceTimeNanos = endTime - startTime,
            outputBuffer = outputTensorBuffer
        )
    }


    fun close() {
        tflite?.close()
    }
}

data class InferenceResult(
    val inferenceTimeNanos: Long,
    val outputBuffer: TensorBuffer?
)
