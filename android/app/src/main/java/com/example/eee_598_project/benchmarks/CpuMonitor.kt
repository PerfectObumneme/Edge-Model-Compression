package com.example.eee_598_project.benchmarks

import java.io.File

class CpuMonitor {
    private val cpuFreqFiles = mutableListOf<File>()
    private var maxFreq = 0L

    init {
        val cpuDirs = File("/sys/devices/system/cpu/").listFiles { file ->
            file.name.startsWith("cpu") && file.name.matches(Regex("cpu\\d+"))
        }

        cpuDirs?.forEach { cpuDir ->
            val freqFile = File(cpuDir, "cpufreq/scaling_cur_freq")
            if (freqFile.exists()) {
                cpuFreqFiles.add(freqFile)
                val maxFreqFile = File(cpuDir, "cpufreq/cpuinfo_max_freq")
                if (maxFreqFile.exists()) {
                    maxFreq = maxOf(maxFreq, maxFreqFile.readText().trim().toLong())
                }
            }
        }
    }

    fun getUsage(): Float {
        var totalUsage = 0f
        cpuFreqFiles.forEach { file ->
            val curFreq = file.readText().trim().toLong()
            totalUsage += curFreq.toFloat() / maxFreq
        }
        return totalUsage / cpuFreqFiles.size
    }
}