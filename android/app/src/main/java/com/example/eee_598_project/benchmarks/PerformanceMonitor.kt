package com.example.eee_598_project.benchmarks

import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.drawable.GradientDrawable
import android.os.BatteryManager
import android.util.Log
import com.github.mikephil.charting.data.Entry
import com.github.mikephil.charting.data.LineData
import com.github.mikephil.charting.data.LineDataSet
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import java.util.concurrent.CopyOnWriteArrayList

class PerformanceMonitor(context: Context) {
    private var startTime: Long = 0
    private val voltageData = CopyOnWriteArrayList<Entry>()
    private val batteryTracker = BatteryTracker(context)
    private var onChartUpdateCallback: ((LineData) -> Unit)? = null

    fun startMonitoring(onChartUpdate: (LineData) -> Unit) {
        startTime = System.currentTimeMillis()
        voltageData.clear()
        onChartUpdateCallback = onChartUpdate

        batteryTracker.startTracking(
            onBatteryUpdate = { timestamp, batteryLevel ->
                val elapsedTime = (timestamp - startTime) / 1000f
                val entry = Entry(elapsedTime, batteryLevel.toFloat())
                voltageData.add(entry)

                // Update chart on the main thread
                CoroutineScope(Dispatchers.Main).launch {
                    updateChart()
                }
            }
        )
    }

    private fun updateChart() {
        val dataSet = LineDataSet(voltageData.toList(), "mAh").apply {
            color = android.graphics.Color.rgb(20,121,122)
            setDrawCircles(false)
            setDrawCircleHole(false)
            setDrawValues(false)
            lineWidth = 2f

            val gradientDrawable = GradientDrawable(
                GradientDrawable.Orientation.TOP_BOTTOM,
                intArrayOf(
                    android.graphics.Color.rgb(21,102,132),
                    android.graphics.Color.argb(50,20,121,122)
                )
            )
            gradientDrawable.alpha = 70
            fillDrawable = gradientDrawable
            setDrawFilled(true)
        }

        val lineData = LineData(dataSet)
        onChartUpdateCallback?.invoke(lineData)
    }

    fun stopMonitoring(): PerfMetrics {
        val endTime = System.currentTimeMillis()
        batteryTracker.stopTracking()

        val energyUsage = calculateEnergyUsage()
        return PerfMetrics(
            energyUsage = energyUsage,
            duration = (endTime - startTime).toDouble(),
            batteryUsageData = voltageData.map { Pair(it.x.toLong(), it.y.toInt()) }
        )
    }

    private fun calculateEnergyUsage(): Double {
        return if (voltageData.size >= 2) {
            val startLevel = voltageData.first().y
            val endLevel = voltageData.last().y
            (startLevel - endLevel).toDouble()
        } else 0.0
    }
}

class BatteryTracker(private val context: Context) {
    private var isTracking = false
    private var trackingThread: Thread? = null
    private var onBatteryUpdateCallback: ((Long, Int) -> Unit)? = null

    fun startTracking(onBatteryUpdate: (Long, Int) -> Unit) {
        isTracking = true
        onBatteryUpdateCallback = onBatteryUpdate

        trackingThread = Thread {
            while (isTracking) {
                val batteryStatus = context.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
                val voltage = batteryStatus?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, -1) ?: -1

                onBatteryUpdateCallback?.invoke(System.currentTimeMillis(), voltage)

                try {
                    Thread.sleep(1000)
                } catch (e: InterruptedException) {
                    Log.e("BatteryTracker", "Tracking interrupted", e)
                    break
                }
            }
        }.apply { start() }
    }

    fun stopTracking() {
        isTracking = false
        trackingThread?.interrupt()
        trackingThread?.join()
    }
}

data class PerfMetrics(
    val energyUsage: Double,
    val duration: Double,
    val batteryUsageData: List<Pair<Long, Int>>
)