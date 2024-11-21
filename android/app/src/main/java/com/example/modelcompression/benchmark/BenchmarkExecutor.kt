class BenchmarkExecutor(private val context: Context) {
    private val modelRunner = ModelRunner(context)
    private val performanceMonitor = PerformanceMonitor(context)
    
    fun runBenchmark(
        numWarmupRuns: Int = 3,
        numBenchmarkRuns: Int = 100,
        socType: ModelRunner.SocType = ModelRunner.SocType.DEFAULT
    ): BenchmarkResults 
    {
        modelRunner.initializeModel(
            useGpu = true,
            useNnapi = true,
            socType = socType
        )
        
        // Prepare input data
        val input = prepareInputData()
        
        // Warmup runs
        repeat(numWarmupRuns) {
            modelRunner.runInference(input)
        }
        
        // Benchmark runs
        val results = mutableListOf<BenchmarkMetrics>()
        
        repeat(numBenchmarkRuns) {
            performanceMonitor.startMonitoring()
            val inferenceResult = modelRunner.runInference(input)
            val perfMetrics = performanceMonitor.stopMonitoring()
            
            results.add(BenchmarkMetrics(
                inferenceTime = inferenceResult.inferenceTimeNanos,
                powerMetrics = perfMetrics.powerMetrics,
                performanceMetrics = perfMetrics.performanceMetrics
            ))
        }
        
        modelRunner.close()
        return BenchmarkResults(results)
    }
    
    private fun prepareInputData(): ByteBuffer {
        // Prepare input based on your model requirements
        return ByteBuffer.allocateDirect(1 * 224 * 224 * 3 * 4).apply {
            order(ByteOrder.nativeOrder())
            // Fill with sample data
            rewind()
        }
    }
}