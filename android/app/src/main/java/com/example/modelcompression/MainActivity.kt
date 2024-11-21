class MainActivity : AppCompatActivity() {
    private lateinit var benchmarkExecutor: BenchmarkExecutor
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        
        benchmarkExecutor = BenchmarkExecutor(this)
        
        // Add UI elements to select SoC type
        val socTypeSpinner: Spinner = findViewById(R.id.socTypeSpinner)
        socTypeSpinner.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            ModelRunner.SocType.values()
        )
        
        findViewById<Button>(R.id.runBenchmarkButton).setOnClickListener {
            lifecycleScope.launch {
                val selectedSocType = socTypeSpinner.selectedItem as ModelRunner.SocType
                runBenchmark(selectedSocType)
            }
        }
    }
    
    private suspend fun runBenchmark(socType: ModelRunner.SocType) {
        withContext(Dispatchers.Default) {
            val results = benchmarkExecutor.runBenchmark(socType = socType)
            val report = BenchmarkReporter.generateReport(results)
            
            // Save results
            FileUtils.saveResults(this@MainActivity, report)
            
            // Update UI
            withContext(Dispatchers.Main) {
                updateResultsUI(report)
            }
        }
    }
}