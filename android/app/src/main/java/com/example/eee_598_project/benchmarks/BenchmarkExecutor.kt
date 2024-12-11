package com.example.eee_598_project.benchmarks

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import org.tensorflow.lite.support.image.TensorImage
import java.io.BufferedInputStream
import java.io.IOException

class CIFAR10DatasetLoader(private val context: Context) {
    companion object {
        private const val DATASET_PATH = "cifar-10-batches-py"
        private const val IMAGE_WIDTH = 32
        private const val IMAGE_HEIGHT = 32
    }

    private val labels = listOf(
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    )

    fun loadTestDataset(): List<Pair<Bitmap, String>> {
        return loadBatchFile()
    }

    private fun loadBatchFile(): List<Pair<Bitmap, String>> {
        val datasetImages = mutableListOf<Pair<Bitmap, String>>()
        val filename = "test_batch"
        try {
            // Open the batch file from assets
            val inputStream = context.assets.open("$DATASET_PATH/$filename")
            val bufferedInputStream = BufferedInputStream(inputStream)

            // Read the entire file
            val fileBytes = bufferedInputStream.readBytes()
            bufferedInputStream.close()

            // Process the binary data
            var offset = 0
            while (offset + 3073 <= fileBytes.size) { // Ensure enough bytes remain
                // Extract label (first byte of each record)
                val labelIndex = fileBytes[offset].toInt() and 0xFF
                val labelName = labels.getOrElse(labelIndex) { "Unknown" }

                // Extract image data
                val imageBitmap = extractImageFromBytes(fileBytes, offset + 1)

                // Add to dataset
                datasetImages.add(imageBitmap to labelName)

                // Move to next record
                offset += 3073
            }

            if (offset < fileBytes.size) {
                Log.w("CIFAR10DatasetLoader", "Remaining ${fileBytes.size - offset} bytes were ignored")
            }


            Log.d("CIFAR10DatasetLoader", "Loaded ${datasetImages.size} images from $filename")
            return datasetImages
        } catch (e: IOException) {
            Log.e("CIFAR10DatasetLoader", "Error loading dataset: ${e.message}")
            return emptyList()
        }
    }

    private fun extractImageFromBytes(bytes: ByteArray, startOffset: Int): Bitmap {
        // Validate if there's enough data for all channels
        if (startOffset + 3 * IMAGE_WIDTH * IMAGE_HEIGHT > bytes.size) {
            throw IllegalArgumentException("Not enough bytes to extract image")
        }

        val bitmap = Bitmap.createBitmap(IMAGE_WIDTH, IMAGE_HEIGHT, Bitmap.Config.ARGB_8888)

        val redChannel = ByteArray(IMAGE_WIDTH * IMAGE_HEIGHT)
        val greenChannel = ByteArray(IMAGE_WIDTH * IMAGE_HEIGHT)
        val blueChannel = ByteArray(IMAGE_WIDTH * IMAGE_HEIGHT)

        // Copy channel data
        System.arraycopy(bytes, startOffset, redChannel, 0, redChannel.size)
        System.arraycopy(bytes, startOffset + IMAGE_WIDTH * IMAGE_HEIGHT, greenChannel, 0, greenChannel.size)
        System.arraycopy(bytes, startOffset + 2 * IMAGE_WIDTH * IMAGE_HEIGHT, blueChannel, 0, blueChannel.size)

        // Reconstruct image
        for (y in 0 until IMAGE_HEIGHT) {
            for (x in 0 until IMAGE_WIDTH) {
                val index = y * IMAGE_WIDTH + x
                val red = redChannel[index].toInt() and 0xFF
                val green = greenChannel[index].toInt() and 0xFF
                val blue = blueChannel[index].toInt() and 0xFF

                val color = Color.rgb(red, green, blue)
                bitmap.setPixel(x, y, color)
            }
        }

        return bitmap
    }

}

class BenchmarkExecutor(context: Context) {
    private val modelRunner = ModelRunner(context)
    private val modelEvaluator = ModelEvaluator(context)
    val performanceMonitor = PerformanceMonitor(context)
    private val cpuMonitor = CpuMonitor()
    private val datasetLoader = CIFAR10DatasetLoader(context)

    fun runBenchmark(modelName: String, onProgressUpdate: (Float) -> Unit): BenchmarkResults {
        modelRunner.initializeModel(modelName)
        modelEvaluator.loadModel(modelName)

        // Load CIFAR-10 test dataset
        val cifar10Dataset = datasetLoader.loadTestDataset().subList(0,10000)
        val totalImages = cifar10Dataset.size
        var processedImages = 0

        val inferenceTimeList = mutableListOf<Long>()
        val cpuUsageList = mutableListOf<Float>()
        val totalRuns = 1

        repeat(totalRuns) {
            cifar10Dataset.forEach { (bitmap, _) ->
                val tensorImage = TensorImage.fromBitmap(bitmap)
                val inferenceResult = modelRunner.runInference(tensorImage)
                inferenceTimeList.add(inferenceResult.inferenceTimeNanos)
                cpuUsageList.add(cpuMonitor.getUsage())

                // Increment processed images and update progress
                processedImages++
                val progress = (processedImages.toFloat() * 100) / totalImages.toFloat()
                onProgressUpdate(progress)
            }
        }

        val accuracyResult = modelEvaluator.evaluateModel(cifar10Dataset)

        val perfMetrics = performanceMonitor.stopMonitoring()
        modelRunner.close()
        modelEvaluator.close()

        val totalInferenceTime = inferenceTimeList.sum() / 1_000_000_000
        val averageCpuUsage = cpuUsageList.average()

        return BenchmarkResults(
            totalInferenceTime = totalInferenceTime,
            totalEnergyUsage = perfMetrics.energyUsage,
            cpuUsage = averageCpuUsage,
            accuracy = accuracyResult.accuracy,
            accuracyDetails = accuracyResult
        )
    }
}

data class BenchmarkResults(
    val totalInferenceTime: Long,
    val totalEnergyUsage: Double,
    val cpuUsage: Double,
    val accuracy: Float,
    val accuracyDetails: ModelAccuracyResult
)