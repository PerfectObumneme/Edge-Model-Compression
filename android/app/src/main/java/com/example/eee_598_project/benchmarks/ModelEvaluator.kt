package com.example.eee_598_project.benchmarks

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException

// ModelEvaluator also needs modification to handle the specific input requirements of TensorFlow Lite models
class ModelEvaluator(private val context: Context) {
    private lateinit var tflite: Interpreter
    private lateinit var labelList: List<String>

    fun loadModel(modelFileName: String) {
        try {
            val modelFile = FileUtil.loadMappedFile(context, modelFileName)
            val options = Interpreter.Options()
            tflite = Interpreter(modelFile, options)

            // Load labels
            val labelFileName = modelFileName.replace(".tflite", "_labels.txt")
            labelList = try {
                context.assets.open(labelFileName).bufferedReader().useLines { it.toList() }
            } catch (e: IOException) {
                Log.w("ModelEvaluator", "Could not load labels: ${e.message}")
                labels
            }
        } catch (e: IOException) {
            Log.e("ModelEvaluator", "Error loading model: ${e.message}")
            throw e
        }
    }

    fun evaluateModel(datasetImages: List<Pair<Bitmap, String>>): ModelAccuracyResult {
        val correctPredictions = datasetImages.count { (bitmap, trueLabel) ->
            val prediction = runSingleInference(bitmap)
            prediction == trueLabel
        }

        val accuracy = correctPredictions.toFloat() / datasetImages.size * 100

        return ModelAccuracyResult(
            totalSamples = datasetImages.size,
            correctPredictions = correctPredictions,
            accuracy = accuracy
        )
    }

    private fun runSingleInference(bitmap: Bitmap): String {
        // Prepare input tensor
        val inputTensor = tflite.getInputTensor(0)
        val inputShape = inputTensor.shape()

        // Create image processor to resize and normalize
        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(inputShape[1], inputShape[2], ResizeOp.ResizeMethod.BILINEAR))
            .add(NormalizeOp(0f, 255f))
            .build()

        val tensorImage = TensorImage.fromBitmap(bitmap)
        val processedImage = imageProcessor.process(tensorImage)

        // Prepare output tensor
        val outputTensor = tflite.getOutputTensor(0)
        val outputBuffer = TensorBuffer.createFixedSize(outputTensor.shape(), outputTensor.dataType())

        // Run inference
        tflite.run(processedImage.buffer, outputBuffer.buffer)

        // Get top prediction
        val probabilities = outputBuffer.floatArray
        val topPredictionIndex = probabilities.indexOfMax() ?: -1

        return if (topPredictionIndex != -1) {
            labelList.getOrElse(topPredictionIndex) { "Unknown" }
        } else {
            "Unknown"
        }
    }

    fun close() {
        tflite.close()
    }

    // Extension function to find index of max value
    private fun FloatArray.indexOfMax(): Int? {
        return indices.maxByOrNull { this[it] }
    }

    companion object {
        private val labels = listOf(
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        )
    }
}

data class ModelAccuracyResult(
    val totalSamples: Int,
    val correctPredictions: Int,
    val accuracy: Float
)
