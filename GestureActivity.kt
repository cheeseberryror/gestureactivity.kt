package com.example.myfslapplication

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.AttributeSet
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ProgressBar
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.facelandmarker.FaceLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt
import org.json.JSONObject
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import org.json.JSONArray

data class Point3D(val x: Float, val y: Float, val z: Float = 0f)
data class GesturePrediction(val gesture: String, val confidence: Float)
data class ShoulderData(val left: Point3D, val right: Point3D, val midpoint: Point3D, val distance: Float)
data class FaceData(val eyeData: EyeData)
data class EyeData(val left: Point3D, val right: Point3D, val midpoint: Point3D, val distance: Float)
data class LandmarkResults(
    val hands: List<List<Point3D>> = emptyList(),
    val pose: List<Point3D> = emptyList(),
    val face: List<Point3D> = emptyList(),
    val faceData: FaceData? = null,
    val shoulderData: ShoulderData? = null,
    val timestamp: Long = System.currentTimeMillis()
)
data class SingleFrameGesture(val name: String, val outputWord: String)
data class SequenceOption(val name: String, val output: String, val nextOptions: List<SequenceOption> = emptyList())
data class Sequence(val starter: String, val starterOutput: String, val options: List<SequenceOption>, val canStandalone: Boolean)
data class MultiStepPhrase(val gestures: List<String>, val output: String)
data class SimilarGestureGroup(val name: String, val gestures: List<String>, val waitDuration: Double = 3.0)
data class GestureConfig(
    val singleFrame: List<SingleFrameGesture>,
    val sequences: List<Sequence>,
    val multiStepPhrases: List<MultiStepPhrase>,
    val similarGestureGroups: List<SimilarGestureGroup>
)

class GestureActivity : AppCompatActivity(), TextToSpeech.OnInitListener {
    private lateinit var previewView: PreviewView
    private lateinit var tvInstructions: TextView
    private lateinit var overlayView: GestureOverlayView
    private lateinit var tvPrediction: TextView
    private lateinit var tvConfidence: TextView
    private lateinit var tvSentence: TextView
    private lateinit var progressHold: ProgressBar
    private lateinit var btnSpace: Button
    private lateinit var btnSpeak: Button
    private lateinit var btnClear: Button
    private lateinit var tvHandCount: TextView
    private lateinit var tvHand1Landmarks: TextView
    private lateinit var tvHand2Landmarks: TextView
    private lateinit var tvFaceLandmarks: TextView
    private lateinit var tvShoulderLandmarks: TextView
    private lateinit var tvFeatureInfo: TextView
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var landmarkDetector: MediaPipeLandmarkDetector
    private lateinit var gesturePipeline: GestureRecognitionPipeline
    private var tts: TextToSpeech? = null
    private var isProcessing = false
    private var frameCount = 0

    companion object {
        private const val TAG = "GestureActivity"
        private const val PERMISSION_REQUEST = 100
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_gesture)
        Log.d(TAG, "========== APP START ==========")
        try {
            initializeUI()
            initializeEngine()
            checkPermissions()
        } catch (e: Exception) {
            Log.e(TAG, "Fatal Error: ${e.message}", e)
            showErrorToast("Init Error: ${e.message}")
        }
    }

    private fun initializeUI() {
        Log.d(TAG, "Initializing UI...")
        previewView = findViewById(R.id.previewView)
        overlayView = findViewById(R.id.overlayView)
        tvPrediction = findViewById(R.id.tvPrediction)
        tvConfidence = findViewById(R.id.tvConfidence)
        tvSentence = findViewById(R.id.tvSentence)
        progressHold = findViewById(R.id.progressHold)
        btnSpace = findViewById(R.id.btnSpace)
        btnSpeak = findViewById(R.id.btnSpeak)
        btnClear = findViewById(R.id.btnClear)
        tvHandCount = findViewById(R.id.tvHandCount)
        tvHand1Landmarks = findViewById(R.id.tvHand1Landmarks)
        tvHand2Landmarks = findViewById(R.id.tvHand2Landmarks)
        tvFaceLandmarks = findViewById(R.id.tvFaceLandmarks)
        tvShoulderLandmarks = findViewById(R.id.tvShoulderLandmarks)
        tvFeatureInfo = findViewById(R.id.tvFeatureInfo)
        tvInstructions = findViewById(R.id.tvInstructions)
        tvPrediction.text = "Initializing..."
        tvConfidence.text = ""

        btnSpace.setOnClickListener {
            Log.d(TAG, "=== SPACE BUTTON PRESSED ===")
            val wordsToAdd = when {
                gesturePipeline.currentProgress.isNotEmpty() -> gesturePipeline.currentMultiStep?.output
                gesturePipeline.pendingWords.isNotEmpty() -> gesturePipeline.pendingWords.joinToString(" ")
                else -> null
            }
            if (wordsToAdd != null && wordsToAdd.isNotEmpty()) {
                gesturePipeline.displayText = if (gesturePipeline.displayText.isEmpty()) {
                    wordsToAdd
                } else {
                    "${gesturePipeline.displayText} $wordsToAdd"
                }
            }
            gesturePipeline.pendingWords.clear()
            gesturePipeline.resetGestureState(fullReset = false)
            tvSentence.text = gesturePipeline.displayText
            tvPrediction.text = "Ready"
            tvInstructions.text = "Perform a gesture to begin"
            overlayView.clear()
        }

        btnClear.setOnClickListener {
            gesturePipeline.resetGestureState(fullReset = true)
            gesturePipeline.displayText = ""
            tvSentence.text = ""
            tvPrediction.text = "Ready"
            tvConfidence.text = ""
            tvInstructions.text = "Perform a gesture to begin"
            overlayView.clear()
        }

        btnSpeak.setOnClickListener {
            val text = tvSentence.text.toString()
            if (text.isNotEmpty()) {
                tts?.speak(text, TextToSpeech.QUEUE_FLUSH, null, "FSL_TTS")
            } else {
                showErrorToast("No text to speak")
            }
        }
    }

    private fun initializeEngine() {
        landmarkDetector = MediaPipeLandmarkDetector(this)
        gesturePipeline = GestureRecognitionPipeline(this)
        tts = TextToSpeech(this, this)
        cameraExecutor = Executors.newSingleThreadExecutor()
    }

    private fun checkPermissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), PERMISSION_REQUEST)
        }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == PERMISSION_REQUEST && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            startCamera()
        }
    }

    private fun startCamera() {
        ProcessCameraProvider.getInstance(this).addListener({
            val provider = ProcessCameraProvider.getInstance(this).get()
            val preview = Preview.Builder().build().apply { setSurfaceProvider(previewView.surfaceProvider) }
            val analyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build().apply { setAnalyzer(cameraExecutor, FrameAnalyzer()) }
            provider.unbindAll()
            provider.bindToLifecycle(this, CameraSelector.DEFAULT_FRONT_CAMERA, preview, analyzer)
            runOnUiThread { tvPrediction.text = "Ready" }
        }, ContextCompat.getMainExecutor(this))
    }

    private inner class FrameAnalyzer : ImageAnalysis.Analyzer {
        override fun analyze(imageProxy: ImageProxy) {
            if (isProcessing) {
                imageProxy.close()
                return
            }
            isProcessing = true
            frameCount++
            try {
                val bitmap = imageProxy.toBitmap() ?: return
                val rotated = rotateBitmapForPhone(bitmap, imageProxy.imageInfo.rotationDegrees)
                val mirrored = mirrorBitmap(rotated)
                val results = landmarkDetector.detectAllLandmarks(mirrored)
                var displayGesture = "No Hand"
                var confidence = 0f
                if (results.hands.isNotEmpty()) {
                    val features = FeatureExtractor.buildFeatureVector(results.hands, results.faceData, results.shoulderData)
                    if (FeatureExtractor.validateFeatureVector(features)) {
                        val prediction = gesturePipeline.processFrame(features)
                        displayGesture = prediction.gesture
                        confidence = prediction.confidence
                    }
                }
                runOnUiThread {
                    val isMultiStepActive = gesturePipeline.currentProgress.isNotEmpty()
                    val isSequenceActive = gesturePipeline.gestureSequence.isNotEmpty()

                    // If a multi-step or sequence is active, only display relevant status
                    if (isMultiStepActive || isSequenceActive) {
                        val currentPredictionName = if (isMultiStepActive) {
                            gesturePipeline.currentMultiStep?.output ?: "..."
                        } else {
                            // For an active sequence, we show the starter output or "Awaiting option"
                            gesturePipeline.gestureConfig?.sequences?.find { it.starter == gesturePipeline.gestureSequence.last() }?.starterOutput ?: "Awaiting option"
                        }

                        val optionsList = gesturePipeline.pendingWords.joinToString(", ")
                        val pendingWordsText = if (gesturePipeline.pendingWords.isNotEmpty()) " [$optionsList]" else ""

                        tvPrediction.text = if (isMultiStepActive) {
                            "Possible: $currentPredictionName ${gesturePipeline.currentProgress.size}/${gesturePipeline.totalSteps}$pendingWordsText"
                        } else {
                            "$currentPredictionName$pendingWordsText"
                        }

                        // Update instructions to prompt for the next step or space bar
                        tvInstructions.text = if (pendingWordsText.isNotEmpty()) {
                            "Space to register or do gesture to continue"
                        } else {
                            "Perform the next gesture in sequence"
                        }

                    } else {
                        // Original logic for when no multi-step or sequence is active
                        val pendingWordsText = if (gesturePipeline.pendingWords.isNotEmpty()) {
                            " [${gesturePipeline.pendingWords.joinToString(", ")}]"
                        } else ""
                        tvPrediction.text = "${gesturePipeline.stableGesture ?: displayGesture}$pendingWordsText"
                        tvInstructions.text = "Perform a gesture to begin"
                    }

                    tvSentence.text = gesturePipeline.displayText
                    tvConfidence.text = if (confidence > 0) String.format("%.0f%%", confidence * 100) else ""
                    progressHold.visibility = if (gesturePipeline.stableGesture == "...") View.VISIBLE else View.INVISIBLE
                    tvPrediction.setTextColor(if (gesturePipeline.stableGesture != null) Color.YELLOW else Color.WHITE)

                    overlayView.updateLandmarks(results.hands, results.pose, results.face, mirrored.width, mirrored.height)
                    overlayView.updateGestureLabel(displayGesture)
                    updateDebugLandmarks(results)
                }

            } finally {
                imageProxy.close()
                isProcessing = false
            }
        }

    private fun rotateBitmapForPhone(bitmap: Bitmap, degrees: Int): Bitmap {
            val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
            return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        }

        private fun mirrorBitmap(bitmap: Bitmap): Bitmap {
            val matrix = Matrix().apply {
                postScale(-1f, 1f)
                postTranslate(bitmap.width.toFloat(), 0f)
            }
            return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        }
    }

    override fun onInit(status: Int) {
        if (status == TextToSpeech.SUCCESS) tts?.language = Locale.US
    }

    private fun showErrorToast(msg: String) {
        runOnUiThread { Toast.makeText(this, msg, Toast.LENGTH_LONG).show() }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        landmarkDetector.cleanup()
        tts?.shutdown()
        gesturePipeline.cleanup()
    }

    override fun onPause() {
        super.onPause()
    }

    override fun onResume() {
        super.onResume()
    }

    private fun updateDebugLandmarks(results: LandmarkResults) {
        tvHandCount.text = "Hands: ${results.hands.size}"
        if (results.hands.isNotEmpty()) {
            tvHand1Landmarks.text = "Hand 1"
        } else {
            tvHand1Landmarks.text = ""
        }
        if (results.hands.size > 1) tvHand2Landmarks.text = "Hand 2" else tvHand2Landmarks.text = ""
        tvFaceLandmarks.text = if (results.faceData?.eyeData != null) "Eyes: ${String.format("%.3f", results.faceData.eyeData.distance)}" else "Eyes: None"
        tvShoulderLandmarks.text = if (results.shoulderData != null) "Shoulders: ${String.format("%.3f", results.shoulderData.distance)}" else "Shoulders: None"
        if (results.hands.isNotEmpty()) {
            val features = FeatureExtractor.buildFeatureVector(results.hands, results.faceData, results.shoulderData)
            tvFeatureInfo.text = "Features: ${features.size}/142"
        } else {
            tvFeatureInfo.text = "Features: 0/142"
        }
    }
}

class GestureRecognitionPipeline(private val context: Context) {
    companion object {
        private const val TAG = "GesturePipeline"
        private const val STABILITY_THRESHOLD = 3
        private const val CONFIDENCE_THRESHOLD = 0.40f
    }

    var displayText = ""
    val pendingWords = mutableListOf<String>()
    val currentProgress: List<String> get() = multiStepProgress
    val totalSteps: Int get() = gestureConfig?.multiStepPhrases?.find {
        it.gestures.contains(multiStepProgress.firstOrNull() ?: "")
    }?.gestures?.size ?: 0
    val gestureSequence = mutableListOf<String>()
    private val frameBuffer = mutableListOf<String>()
    private var classifier: GestureClassifier? = null
    private var featureScaler: FeatureScaler? = null
    private var isInitialized = false
    private var frameCount = 0
    var gestureConfig: GestureConfig? = null
    var currentMultiStep: MultiStepPhrase? = null
    private var multiStepProgress = mutableListOf<String>()
    private var gestureLocked = false
    var stableGesture: String? = null
    private var stableGestureCount = 0

    init {
        initializePipeline()
    }

    private fun initializePipeline() {
        try {
            val labels = loadLabelsFromAssets()
            val modelBytes = context.assets.open("gesture_model.tflite").readBytes()
            classifier = GestureClassifier(context, modelBytes, labels)
            featureScaler = FeatureScaler(context)
            gestureConfig = loadGestureConfig()
            isInitialized = true
        } catch (e: Exception) {
            Log.e(TAG, "❌ Error: ${e.message}", e)
        }
    }

    private fun loadGestureConfig(): GestureConfig? {
        return try {
            val configString = context.assets.open("gesture_config.json").bufferedReader().use { it.readText() }
            val json = JSONObject(configString)
            val configJson = json.getJSONObject("gesture_classes")
            val singleFrameList = mutableListOf<SingleFrameGesture>()
            val sfArray = configJson.getJSONArray("single_frame")
            for (i in 0 until sfArray.length()) {
                val obj = sfArray.getJSONObject(i)
                singleFrameList.add(SingleFrameGesture(obj.getString("name"), obj.getString("output_word")))
            }
            val sequencesList = mutableListOf<Sequence>()
            val seqArray = configJson.getJSONArray("sequences")
            for (i in 0 until seqArray.length()) {
                val obj = seqArray.getJSONObject(i)
                val options = parseSequenceOptions(obj.getJSONArray("options"))
                sequencesList.add(Sequence(obj.getString("starter"), obj.getString("starter_output"), options, obj.optBoolean("can_standalone", false)))
            }
            val multiStepList = mutableListOf<MultiStepPhrase>()
            val msArray = configJson.optJSONArray("multi_step_phrases")
            if (msArray != null) {
                for (i in 0 until msArray.length()) {
                    val obj = msArray.getJSONObject(i)
                    val gestures = mutableListOf<String>()
                    val gesturesArray = obj.getJSONArray("gestures")
                    for (j in 0 until gesturesArray.length()) {
                        gestures.add(gesturesArray.getString(j))
                    }
                    multiStepList.add(MultiStepPhrase(gestures, obj.getString("output")))
                }
            }
            GestureConfig(singleFrameList, sequencesList, multiStepList, emptyList())
        } catch (e: Exception) {
            Log.e(TAG, "Error loading config: ${e.message}")
            null
        }
    }

    private fun parseSequenceOptions(array: JSONArray): List<SequenceOption> {
        val options = mutableListOf<SequenceOption>()
        for (i in 0 until array.length()) {
            val obj = array.getJSONObject(i)
            options.add(SequenceOption(obj.getString("name"), obj.getString("output")))
        }
        return options
    }

    fun processFrame(features: FloatArray): GesturePrediction {
        if (!isInitialized || featureScaler == null || classifier == null) {
            return GesturePrediction("Init...", 0f)
        }
        if (!FeatureExtractor.validateFeatureVector(features)) {
            return GesturePrediction("None", 0f)
        }
        val scaledFeatures = featureScaler!!.scaleFeatures(features)
        val (prediction, confidence) = classifier!!.predict(scaledFeatures)
        frameCount++
        if (prediction == "None" || prediction == "No Hand") {
            if (frameCount % 30 == 0) {
                resetGestureState(fullReset = false)
            }
        }

        // RESTRICTION: Block sequence options unless correct starter is active
        val isSequenceOption = gestureConfig?.sequences?.any { seq ->
            seq.options.any { it.name == prediction }
        } ?: false

        if (isSequenceOption) {
            val hasCorrectStarter = gestureSequence.isNotEmpty() && gestureConfig?.sequences?.any { seq ->
                seq.starter == gestureSequence.last() && seq.options.any { it.name == prediction }
            } ?: false
            if (!hasCorrectStarter) {
                return GesturePrediction("None", 0f)
            }
        }

        // RESTRICTION: Block multi-step Step 2+ gestures
        val isRestrictedMultiStep = gestureConfig?.multiStepPhrases?.any { phrase ->
            val index = phrase.gestures.indexOf(prediction)
            index > 0
        } ?: false

        if (isRestrictedMultiStep) {
            val isCorrectNextStep = multiStepProgress.isNotEmpty() &&
                    currentMultiStep?.gestures?.getOrNull(multiStepProgress.size) == prediction
            if (!isCorrectNextStep) {
                return GesturePrediction("None", 0f)
            }
        }

        if (prediction == "error") return GesturePrediction("error", 0f)

        val stablePrediction = applyStabilityFilter(prediction, confidence)
        if (stablePrediction != "..." && stablePrediction != "None" && confidence >= CONFIDENCE_THRESHOLD && !gestureLocked) {
            stableGesture = stablePrediction
            handleGestureDetection(stablePrediction)
        }
        return GesturePrediction(stablePrediction, confidence)
    }

    private fun handleGestureDetection (gestureName: String) {
        if (gestureLocked) return //
        val config = gestureConfig?: return //

        // CRITICAL FIX: If an active sequence or multi-step phrase exists,
        // only proceed if the current gesture is explicitly the correct next step or a control gesture.
        // Otherwise, return immediately to ignore the invalid gesture.

        // 1. HANDLE ACTIVE SEQUENCES
        if (gestureSequence.isNotEmpty()) { //
            val lastStarter = gestureSequence.last() //
            val sequence = config.sequences.find { it.starter == lastStarter } //
            val option = sequence?.options?.find { it.name == gestureName } //

            if (option != null) {
                // Correct option detected, process it
                val combinedOutput = "${sequence.starterOutput} ${option.output}" //
                emitWord(combinedOutput) //
                completeGestureCycle() //
                return // Exit after handling a valid sequence option
            } else if (gestureName == lastStarter) {
                // The user repeated the starter gesture, ignore it this frame
                return
            } else if (gestureName != "..." && gestureName != "None") {
                // An invalid, but otherwise "valid" gesture was performed during an active sequence.
                // Ignore it this frame, do not reset the state unless it times out or is cleared manually.
                return
            }
        }

        // 2. HANDLE MULTI-STEP PHRASES
        if (multiStepProgress.isNotEmpty()) { //
            val expected = currentMultiStep?.gestures?.getOrNull(multiStepProgress.size) //

            if (gestureName == expected) {
                // Correct next step detected, process it
                multiStepProgress.add(gestureName) //
                if (multiStepProgress.size == currentMultiStep?.gestures?.size) {
                    commitPendingToDisplay() //
                    completeGestureCycle() //
                }
                return // Exit after processing a multi-step component
            } else if (gestureName != "..." && gestureName != "None") {
                // An invalid, but otherwise "valid" gesture was performed during an active multi-step phrase.
                // Ignore it this frame, do not reset the state unless it times out or is cleared manually.
                // The previous logic here called resetGestureState(), which was the primary bug.
                return
            }
        }

        // 3. START NEW SEQUENCES OR STANDALONE GESTURES
        // This block runs only if no sequence or multi-step phrase is active (due to the 'return' statements above).
        // START NEW MULTI-STEP PHRASES
        val multiStepStarter = config.multiStepPhrases.find { //
            it.gestures.firstOrNull() == gestureName
        }
        if (multiStepStarter != null) { //
            currentMultiStep = multiStepStarter
            multiStepProgress.add(gestureName)
            lockGestureTemporarily() //
            return // Exit after starting a new multi-step phrase
        }

        // START NEW SEQUENCES
        val starterSeq = config.sequences.find { it.starter == gestureName } //
        if (starterSeq != null) { //
            gestureSequence.add(gestureName) //
            emitWord(starterSeq.starterOutput) //
            lockGestureTemporarily() //
            return // Exit after starting a new sequence
        }

        // STANDALONE SINGLE FRAME
        val singleFrame = config.singleFrame.find { it.name == gestureName } //
        if (singleFrame != null) { //
            emitWord(singleFrame.outputWord) //
            commitPendingToDisplay() //
            completeGestureCycle() //
            return // Exit after processing a standalone gesture
        }
    }

    private fun completeGestureCycle() {
        lockGestureTemporarily()
    }

    private fun commitPendingToDisplay() {
        if (pendingWords.isNotEmpty()) {
            val wordsToAdd = pendingWords.joinToString(" ")
            displayText = if (displayText.isEmpty()) wordsToAdd else "$displayText $wordsToAdd"
            pendingWords.clear()
        }
    }

    private fun emitWord(word: String) {
        if (pendingWords.isEmpty() || pendingWords.last() != word) {
            pendingWords.add(word)
        }
    }

    private fun lockGestureTemporarily() {
        gestureLocked = true
        android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
            gestureLocked = false
            stableGesture = null
        }, 800)
    }

    private fun applyStabilityFilter(currentPrediction: String, confidence: Float): String {
        if (confidence < CONFIDENCE_THRESHOLD) return "None"
        if (currentPrediction == "None" || currentPrediction == "unknown") return "..."
        frameBuffer.add(currentPrediction)
        if (frameBuffer.size > STABILITY_THRESHOLD) frameBuffer.removeAt(0)
        val counts = frameBuffer.groupingBy { it }.eachCount()
        val mostFrequent = counts.maxByOrNull { it.value }
        return if ((mostFrequent?.value ?: 0) >= STABILITY_THRESHOLD) mostFrequent!!.key else "..."
    }

    private fun loadLabelsFromAssets(): List<String> {
        return try {
            context.assets.open("gesture_labels.txt").bufferedReader().use { it.readText() }
                .split("\n").filter { it.isNotBlank() }.map { it.trim() }
        } catch (e: Exception) {
            emptyList()
        }
    }

    fun resetGestureState(fullReset: Boolean = false) {
        if (fullReset) {
            displayText = ""
            pendingWords.clear()
        }
        frameBuffer.clear()
        gestureSequence.clear()
        currentMultiStep = null
        multiStepProgress.clear()
        gestureLocked = false
        stableGesture = null
        stableGestureCount = 0
    }

    fun cleanup() {
        classifier?.cleanup()
    }
}

class GestureClassifier(context: Context, modelBytes: ByteArray, private val labels: List<String>) {
    private var interpreter: Interpreter? = null
    companion object {
        private const val TAG = "GestureClassifier"
        private const val EXPECTED_FEATURES = 142
    }

    init {
        val modelBuffer = ByteBuffer.allocateDirect(modelBytes.size).order(ByteOrder.nativeOrder())
        modelBuffer.put(modelBytes)
        modelBuffer.rewind()
        val options = Interpreter.Options().apply {
            setNumThreads(4)
            setUseNNAPI(true)
        }
        interpreter = Interpreter(modelBuffer, options)
    }

    fun predict(features: FloatArray): Pair<String, Float> {
        val interp = interpreter ?: return "error" to 0f
        if (features.size != EXPECTED_FEATURES) return "error" to 0f
        val inputBuffer = ByteBuffer.allocateDirect(4 * EXPECTED_FEATURES).order(ByteOrder.nativeOrder()).asFloatBuffer()
        inputBuffer.put(features)
        inputBuffer.rewind()
        val outputShape = interp.getOutputTensor(0).shape()
        val outputArray = Array(1) { FloatArray(outputShape[1]) }
        interp.runForMultipleInputsOutputs(arrayOf(inputBuffer), mapOf(0 to outputArray))
        val probabilities = outputArray[0]
        var maxIdx = 0
        var maxProb = probabilities[0]
        for (i in 1 until probabilities.size) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i]
                maxIdx = i
            }
        }
        val predictedLabel = if (maxIdx < labels.size) labels[maxIdx] else "unknown"
        return predictedLabel to maxProb
    }

    fun cleanup() {
        interpreter?.close()
        interpreter = null
    }
}

class FeatureScaler(context: Context) {
    companion object {
        private const val FEATURE_COUNT = 142
    }
    private val mean = FloatArray(FEATURE_COUNT)
    private val scale = FloatArray(FEATURE_COUNT)

    init {
        val jsonString = context.assets.open("scaler_data.json").bufferedReader().use { it.readText() }
        val json = JSONObject(jsonString)
        val meanArray = json.getJSONArray("mean")
        val scaleArray = json.getJSONArray("scale")
        for (i in 0 until FEATURE_COUNT) {
            mean[i] = meanArray.getDouble(i).toFloat()
            scale[i] = scaleArray.getDouble(i).toFloat()
            if (scale[i] == 0f) scale[i] = 1f
        }
    }

    fun scaleFeatures(input: FloatArray): FloatArray {
        val output = FloatArray(FEATURE_COUNT)
        for (i in input.indices) {
            output[i] = (input[i] - mean[i]) / scale[i]
        }
        return output
    }
}

object FeatureExtractor {
    private const val TOTAL_FEATURES = 142
    private const val RELATIVE_FEATURES_PER_HAND = 42
    private const val ROTATION_FEATURES_PER_HAND = 21

    fun buildFeatureVector(hands: List<List<Point3D>>, faceData: FaceData?, shoulderData: ShoulderData?): FloatArray {
        val features = FloatArray(TOTAL_FEATURES) { 0f }
        for (handIndex in 0 until 2) {
            val hand = hands.getOrNull(handIndex)
            if (hand != null && hand.size >= 21) {
                extractRelativeHandFeatures(hand, features, handIndex * 42)
            }
        }
        for (handIndex in 0 until 2) {
            val hand = hands.getOrNull(handIndex)
            if (hand != null && hand.size >= 21) {
                extractRotationFeatures(hand, features, 84 + (handIndex * 21))
            }
        }
        for (handIndex in 0 until 2) {
            val hand = hands.getOrNull(handIndex)
            if (hand != null && hand.size >= 21) {
                extractBodyRelationForHand(hand, faceData, shoulderData, features, 126 + (handIndex * 8))
            }
        }
        return features
    }

    private fun extractRelativeHandFeatures(handLandmarks: List<Point3D>, features: FloatArray, startIdx: Int) {
        if (handLandmarks.size < 21) return
        val wrist = handLandmarks[0]
        var idx = startIdx
        for (i in 0 until 21) {
            val lm = handLandmarks[i]
            features[idx++] = lm.x - wrist.x
            features[idx++] = lm.y - wrist.y
        }
        val maxVal = features.slice(startIdx until startIdx + RELATIVE_FEATURES_PER_HAND).maxOfOrNull { abs(it) } ?: 1f
        if (maxVal > 0) {
            for (i in startIdx until startIdx + RELATIVE_FEATURES_PER_HAND) {
                features[i] /= maxVal
            }
        }
    }

    private fun extractRotationFeatures(handLandmarks: List<Point3D>, features: FloatArray, startIdx: Int) {
        if (handLandmarks.size < 21) return
        var idx = startIdx
        val wrist = handLandmarks[0]
        val middleMCP = handLandmarks[9]
        val indexMCP = handLandmarks[5]
        val pinkyMCP = handLandmarks[17]
        val thumbMCP = handLandmarks[2]
        val thumbTip = handLandmarks[4]
        val rawRotation = atan2(middleMCP.y - wrist.y, middleMCP.x - wrist.x)
        val rawRoll = atan2(pinkyMCP.y - indexMCP.y, pinkyMCP.x - indexMCP.x)
        val rawPitch = atan2(thumbTip.y - thumbMCP.y, thumbTip.x - thumbMCP.x)
        features[idx++] = normalizeAngle(rawRotation)
        features[idx++] = normalizeAngle(rawRoll)
        features[idx++] = normalizeAngle(rawPitch)
        val handWidth = sqrt((pinkyMCP.x - indexMCP.x).pow(2) + (pinkyMCP.y - indexMCP.y).pow(2))
        val palmLength = sqrt((middleMCP.x - wrist.x).pow(2) + (middleMCP.y - wrist.y).pow(2))
        features[idx++] = if (palmLength > 0) handWidth / palmLength else 0f
        val fingerAngles = extractFingerAngles(handLandmarks)
        for (i in 0 until min(17, fingerAngles.size)) {
            features[idx++] = fingerAngles[i]
        }
        while (idx < startIdx + 21) {
            features[idx++] = 0f
        }
    }

    private fun normalizeAngle(angle: Float): Float {
        val pi = Math.PI.toFloat()
        return ((angle + pi) % (2 * pi)) - pi
    }

    private fun extractFingerAngles(handLandmarks: List<Point3D>): FloatArray {
        if (handLandmarks.isEmpty()) return FloatArray(15)
        val angles = mutableListOf<Float>()
        val fingerIndices = listOf(
            listOf(0, 1, 2, 3, 4),
            listOf(0, 5, 6, 7, 8),
            listOf(0, 9, 10, 11, 12),
            listOf(0, 13, 14, 15, 16),
            listOf(0, 17, 18, 19, 20)
        )
        for (finger in fingerIndices) {
            for (i in 0 until finger.size - 2) {
                val p1 = handLandmarks[finger[i]]
                val p2 = handLandmarks[finger[i + 1]]
                val p3 = handLandmarks[finger[i + 2]]
                val v1x = p1.x - p2.x
                val v1y = p1.y - p2.y
                val v2x = p3.x - p2.x
                val v2y = p3.y - p2.y
                val dot = v1x * v2x + v1y * v2y
                val norm1 = sqrt(v1x * v1x + v1y * v1y)
                val norm2 = sqrt(v2x * v2x + v2y * v2y)
                val angle = if (norm1 > 1e-6 && norm2 > 1e-6) {
                    acos((dot / (norm1 * norm2)).coerceIn(-1f, 1f))
                } else 0f
                angles.add(angle)
            }
        }
        return angles.toFloatArray()
    }

    private fun extractBodyRelationForHand(handLandmarks: List<Point3D>, faceData: FaceData?, shoulderData: ShoulderData?, features: FloatArray, startIdx: Int) {
        var idx = startIdx
        val wrist = handLandmarks[0]
        val eyeData = faceData?.eyeData
        if (eyeData != null) {
            val distToEyes = calculateDistance(wrist, eyeData.midpoint)
            features[idx++] = if (eyeData.distance > 0) distToEyes / eyeData.distance else distToEyes
            features[idx++] = wrist.y - eyeData.midpoint.y
            features[idx++] = wrist.x - eyeData.left.x
            features[idx++] = wrist.x - eyeData.right.x
        } else {
            repeat(4) { features[idx++] = 0f }
        }
        if (shoulderData != null) {
            val distToShoulders = calculateDistance(wrist, shoulderData.midpoint)
            features[idx++] = if (shoulderData.distance > 0) distToShoulders / shoulderData.distance else distToShoulders
            features[idx++] = wrist.y - shoulderData.midpoint.y
            features[idx++] = wrist.x - shoulderData.left.x
            features[idx++] = wrist.x - shoulderData.right.x
        } else {
            repeat(4) { features[idx++] = 0f }
        }
    }

    private fun calculateDistance(p1: Point3D, p2: Point3D): Float {
        val dx = p1.x - p2.x
        val dy = p1.y - p2.y
        return sqrt(dx * dx + dy * dy)
    }

    fun validateFeatureVector(features: FloatArray): Boolean {
        if (features.size != TOTAL_FEATURES) return false
        for (i in features.indices) {
            if (features[i].isNaN() || features[i].isInfinite()) return false
        }
        return true
    }
}

class MediaPipeLandmarkDetector(private val context: Context) {
    private var handLandmarker: HandLandmarker? = null
    private var poseLandmarker: PoseLandmarker? = null
    private var faceLandmarker: FaceLandmarker? = null
    companion object {
        private const val TAG = "MediaPipeDetector"
        private val LEFT_EYE_INDICES = listOf(33, 133, 157, 158, 159, 160, 161, 173, 246, 7, 163, 144, 145, 153, 154, 155)
        private val RIGHT_EYE_INDICES = listOf(362, 263, 386, 387, 388, 389, 390, 466, 398, 382, 384, 381, 380, 374, 373, 390)
    }

    init {
        initializeHandLandmarker()
        initializePoseLandmarker()
        initializeFaceLandmarker()
    }

    private fun initializeHandLandmarker() {
        val baseOptions = BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build()
        handLandmarker = HandLandmarker.createFromOptions(context, HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(baseOptions).setNumHands(2).setMinHandDetectionConfidence(0.5f)
            .setMinHandPresenceConfidence(0.5f).setMinTrackingConfidence(0.5f)
            .setRunningMode(com.google.mediapipe.tasks.vision.core.RunningMode.IMAGE).build())
    }

    private fun initializePoseLandmarker() {
        val baseOptions = BaseOptions.builder().setModelAssetPath("pose_landmarker.task").build()
        poseLandmarker = PoseLandmarker.createFromOptions(context, PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(baseOptions).setMinPoseDetectionConfidence(0.5f)
            .setMinPosePresenceConfidence(0.5f).setMinTrackingConfidence(0.5f)
            .setRunningMode(com.google.mediapipe.tasks.vision.core.RunningMode.IMAGE).build())
    }

    private fun initializeFaceLandmarker() {
        val baseOptions = BaseOptions.builder().setModelAssetPath("face_landmarker.task").build()
        faceLandmarker = FaceLandmarker.createFromOptions(context, FaceLandmarker.FaceLandmarkerOptions.builder()
            .setBaseOptions(baseOptions).setNumFaces(1).setMinFaceDetectionConfidence(0.5f)
            .setMinFacePresenceConfidence(0.5f).setMinTrackingConfidence(0.5f)
            .setOutputFaceBlendshapes(false).setOutputFacialTransformationMatrixes(false)
            .setRunningMode(com.google.mediapipe.tasks.vision.core.RunningMode.IMAGE).build())
    }

    fun detectAllLandmarks(bitmap: Bitmap): LandmarkResults {
        val hands = mutableListOf<List<Point3D>>()
        val pose = mutableListOf<Point3D>()
        val face = mutableListOf<Point3D>()
        var faceData: FaceData? = null
        var shoulderData: ShoulderData? = null
        try {
            val mpImage = BitmapImageBuilder(bitmap).build()
            handLandmarker?.detect(mpImage)?.let { handResult ->
                if (handResult.landmarks().isNotEmpty()) {
                    val sortedLandmarks = handResult.landmarks().sortedBy { it[0].x() }
                    for (handLandmarks in sortedLandmarks) {
                        val hand = handLandmarks.map { landmark ->
                            Point3D(landmark.x(), landmark.y(), landmark.z())
                        }
                        hands.add(hand)
                    }
                }
            }
            poseLandmarker?.detect(mpImage)?.let { poseResult ->
                if (poseResult.landmarks().isNotEmpty()) {
                    val landmarks = poseResult.landmarks()[0]
                    landmarks.forEach { landmark ->
                        pose.add(Point3D(landmark.x(), landmark.y(), landmark.z()))
                    }
                    if (pose.size > 12) {
                        val left = pose[11]
                        val right = pose[12]
                        val midpoint = Point3D((left.x + right.x) / 2f, (left.y + right.y) / 2f, (left.z + right.z) / 2f)
                        val shoulderDist = calculateDistance(left, right)
                        shoulderData = ShoulderData(left, right, midpoint, shoulderDist)
                    }
                }
            }
            faceLandmarker?.detect(mpImage)?.let { faceResult ->
                if (faceResult.faceLandmarks().isNotEmpty()) {
                    val faceLandmarks = faceResult.faceLandmarks()[0]
                    faceLandmarks.forEach { landmark ->
                        face.add(Point3D(landmark.x(), landmark.y(), landmark.z()))
                    }
                    faceData = extractEyeDataFromFace(face)
                }
            }
        } catch (e: Exception) {
            Log.e("MediaPipeDetector", "Detect error: ${e.message}", e)
        }
        return LandmarkResults(hands, pose, face, faceData, shoulderData)
    }

    private fun extractEyeDataFromFace(faceLandmarks: List<Point3D>): FaceData? {
        if (faceLandmarks.size < 468) return null
        try {
            var leftX = 0f; var leftY = 0f; var leftZ = 0f
            for (idx in LEFT_EYE_INDICES) {
                val lm = faceLandmarks[idx]
                leftX += lm.x; leftY += lm.y; leftZ += lm.z
            }
            val leftEye = Point3D(leftX / LEFT_EYE_INDICES.size, leftY / LEFT_EYE_INDICES.size, leftZ / LEFT_EYE_INDICES.size)
            var rightX = 0f; var rightY = 0f; var rightZ = 0f
            for (idx in RIGHT_EYE_INDICES) {
                val lm = faceLandmarks[idx]
                rightX += lm.x; rightY += lm.y; rightZ += lm.z
            }
            val rightEye = Point3D(rightX / RIGHT_EYE_INDICES.size, rightY / RIGHT_EYE_INDICES.size, rightZ / RIGHT_EYE_INDICES.size)
            val midpoint = Point3D((leftEye.x + rightEye.x) / 2, (leftEye.y + rightEye.y) / 2, (leftEye.z + rightEye.z) / 2)
            val eyeDistance = calculateDistance(leftEye, rightEye)
            return FaceData(eyeData = EyeData(leftEye, rightEye, midpoint, eyeDistance))
        } catch (e: Exception) {
            return null
        }
    }

    private fun calculateDistance(p1: Point3D, p2: Point3D): Float {
        val dx = p1.x - p2.x
        val dy = p1.y - p2.y
        return sqrt(dx * dx + dy * dy)
    }

    fun cleanup() {
        handLandmarker?.close()
        poseLandmarker?.close()
        faceLandmarker?.close()
    }
}

class GestureOverlayView @JvmOverloads constructor(context: Context, attrs: AttributeSet? = null, defStyleAttr: Int = 0) : View(context, attrs, defStyleAttr) {
    private val handPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 4f
        isAntiAlias = true
    }
    private val landmarkPaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.FILL
        isAntiAlias = true
    }
    private val labelPaint = Paint().apply {
        color = Color.YELLOW
        textSize = 48f
        isFakeBoldText = true
        isAntiAlias = true
    }
    private var handLandmarks: List<List<Point3D>> = emptyList()
    private var poseLandmarks: List<Point3D> = emptyList()
    private var currentGesture: String = ""
    private var imageWidth = 0
    private var imageHeight = 0
    private val handConnections = listOf(
        Pair(0, 1), Pair(1, 2), Pair(2, 3), Pair(3, 4),
        Pair(0, 5), Pair(5, 6), Pair(6, 7), Pair(7, 8),
        Pair(0, 9), Pair(9, 10), Pair(10, 11), Pair(11, 12),
        Pair(0, 13), Pair(13, 14), Pair(14, 15), Pair(15, 16),
        Pair(0, 17), Pair(17, 18), Pair(18, 19), Pair(19, 20),
        Pair(5, 9), Pair(9, 13), Pair(13, 17)
    )

    fun updateLandmarks(hands: List<List<Point3D>>, pose: List<Point3D>, face: List<Point3D>?, imageWidth: Int, imageHeight: Int) {
        this.handLandmarks = hands
        this.poseLandmarks = pose
        this.imageWidth = imageWidth
        this.imageHeight = imageHeight
        postInvalidate()
    }

    fun updateGestureLabel(gesture: String) {
        this.currentGesture = gesture
        postInvalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (imageWidth == 0 || imageHeight == 0) return
        val scaleX = width.toFloat() / imageWidth
        val scaleY = height.toFloat() / imageHeight
        val scale = minOf(scaleX, scaleY)
        canvas.save()
        canvas.translate((width - imageWidth * scale) / 2, (height - imageHeight * scale) / 2)
        canvas.scale(scale, scale)
        for (hand in handLandmarks) {
            if (hand.size >= 21) {
                for ((start, end) in handConnections) {
                    if (start < hand.size && end < hand.size) {
                        canvas.drawLine(hand[start].x * imageWidth, hand[start].y * imageHeight,
                            hand[end].x * imageWidth, hand[end].y * imageHeight, handPaint)
                    }
                }
                for (lm in hand) {
                    canvas.drawCircle(lm.x * imageWidth, lm.y * imageHeight, 4f, landmarkPaint)
                }
            }
        }
        canvas.restore()
        if (currentGesture.isNotEmpty() && currentGesture != "..." && currentGesture != "No Hand" && currentGesture != "None") {
            val textWidth = labelPaint.measureText(currentGesture)
            canvas.drawText(currentGesture, (width - textWidth) / 2, 100f, labelPaint)
        }
    }

    fun clear() {
        handLandmarks = emptyList()
        poseLandmarks = emptyList()
        currentGesture = ""
        postInvalidate()
    }
}