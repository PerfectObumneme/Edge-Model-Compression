class PerformanceMonitor(private val context: Context) {
    private val powerMonitor = PowerMonitor(context)
    private val cpuMonitor = CpuMonitor()
    private val memoryMonitor = MemoryMonitor(context)
    
    fun startMonitoring() {
        powerMonitor.startMonitoring()
        cpuMonitor.startMonitoring()
        memoryMonitor.startMonitoring()
    }
    
    fun stopMonitoring(): PerformanceMetrics {
        return PerformanceMetrics(
            powerMetrics = powerMonitor.stopMonitoring(),
            cpuUsage = cpuMonitor.stopMonitoring(),
            memoryUsage = memoryMonitor.stopMonitoring()
        )
    }
}
