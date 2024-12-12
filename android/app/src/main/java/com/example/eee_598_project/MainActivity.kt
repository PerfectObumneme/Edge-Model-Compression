package com.example.eee_598_project

import android.Manifest
import android.os.Bundle
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.ProgressBar
import android.widget.Spinner
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.lifecycle.lifecycleScope
import com.example.eee_598_project.benchmarks.BenchmarkExecutor
import com.github.mikephil.charting.charts.LineChart
import com.github.mikephil.charting.components.Legend
import com.github.mikephil.charting.data.LineData
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale

class MainActivity : ComponentActivity() {
    private var benchmarkResults: String? = null
    lateinit var progressBar: ProgressBar
    lateinit var progressTextView: TextView
    private lateinit var voltageChart: LineChart

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        progressBar = findViewById(R.id.progressBar)
        progressTextView = findViewById(R.id.progressTextView)
        voltageChart = findViewById(R.id.voltageChart)
        setupModelSpinner()
        setupBenchmarkButton()
        setupChart()

    }

    private fun setupModelSpinner() {
        val modelSpinner: Spinner = findViewById(R.id.modelSpinner)

        val modelList = try {
            assets.list("")?.filter { it.endsWith(".tflite") } ?: emptyList()
        } catch (e: Exception) {
            e.printStackTrace()
            emptyList<String>()
        }

        modelSpinner.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            modelList
        )
    }

    private fun setupBenchmarkButton() {
        val runButton: Button = findViewById(R.id.runBenchmarkButton)
        runButton.setOnClickListener {
            checkAndRequestPermissions()
        }
    }

    private fun checkAndRequestPermissions() {
        arrayOf(
            Manifest.permission.READ_EXTERNAL_STORAGE
        )
        executeBenchmark()
    }

    private fun setupChart() {
        voltageChart.apply {
            description.isEnabled = false
            axisLeft.apply {
                axisMinimum = 0f
                axisMaximum = 5000f
                granularity = 500f
            }
            axisRight.isEnabled = false
            xAxis.apply {
                granularity = 1f
                isGranularityEnabled = true
                position = com.github.mikephil.charting.components.XAxis.XAxisPosition.BOTTOM
            }
            // Enable pinch zooming and scaling
            setPinchZoom(true)
            isScaleXEnabled = true
            isScaleYEnabled = true

            // Enable a clean, modern look
            setDrawGridBackground(false)
            legend.apply {
                form = Legend.LegendForm.LINE
                textSize = 12f
                textColor = android.graphics.Color.DKGRAY
                isEnabled = true
            }
        }
        // Initial empty data
        voltageChart.data = LineData()
        voltageChart.invalidate()
    }

    private fun executeBenchmark() {
        val modelSpinner: Spinner = findViewById(R.id.modelSpinner)
        val resultTextView: TextView = findViewById(R.id.resultTextView)

        val selectedModel = modelSpinner.selectedItem as String
        resultTextView.text = getString(R.string.running_benchmark, selectedModel)

        // Reset the progress bar and chart
        progressBar.progress = 0
        voltageChart.clear()
        voltageChart.data = LineData()
        voltageChart.invalidate()

        // Start a coroutine to execute the benchmark
        lifecycleScope.launch {
            // Call the suspending function runBenchmark inside the coroutine
            benchmarkResults = runBenchmark(selectedModel, progressBar)
            resultTextView.text = benchmarkResults
        }
    }

    private suspend fun runBenchmark(modelName: String, progressBar: ProgressBar): String =
        withContext(Dispatchers.Default) {
            val benchmarkExecutor = BenchmarkExecutor(this@MainActivity)

            // Update the chart periodically
            benchmarkExecutor.performanceMonitor.startMonitoring { lineData ->
                runOnUiThread {
                    voltageChart.data = lineData
                    voltageChart.notifyDataSetChanged()
                    voltageChart.invalidate()
                }
            }

            val results = benchmarkExecutor.runBenchmark(modelName) { progress ->
                // Update the progress bar on the main thread
                progressBar.post {
                    progressBar.progress = progress.toInt()
                    progressTextView.text = String.format(Locale.getDefault(), "%.2f%%", progress.toFloat())
                }
            }

            getString(
                R.string.benchmark_results,
                modelName,
                results.totalInferenceTime.toFloat(),
                results.totalEnergyUsage.toFloat(),
                results.accuracy.toFloat()
            )
        }

}