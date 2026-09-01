/**
 * MindLens XAI — Major-Project Grade Explainable Mental Health Platform
 * Multilingual AI Voice Engine, Real-Time SQLite Analytics, Multi-Model Benchmarks & Clinical PDF Reports
 */

document.addEventListener("DOMContentLoaded", () => {
  // --- Active Voice Language State ---
  let voiceLang = "hi-IN"; // Default language

  // --- DOM References ---
  const statementInput = document.getElementById("statementInput");
  const currentCount = document.getElementById("currentCount");
  const wordCount = document.getElementById("wordCount");
  const clearBtn = document.getElementById("clearBtn");
  const randomizeBtn = document.getElementById("randomizeBtn");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const loadingSpinner = document.getElementById("loadingSpinner");
  const topNWordsRange = document.getElementById("topNWordsRange");
  const topNWordsDisplay = document.getElementById("topNWordsDisplay");
  const presetChipsContainer = document.getElementById("presetChipsContainer");
  const modelSelectDropdown = document.getElementById("modelSelectDropdown");
  const activeModelBadge = document.getElementById("activeModelBadge");
  const bestModelPill = document.getElementById("bestModelPill");
  const modelComparisonPills = document.getElementById("modelComparisonPills");

  // User Accuracy Feedback & Modal Elements
  const userFeedbackSection = document.getElementById("userFeedbackSection");
  const feedbackAccurateBtn = document.getElementById("feedbackAccurateBtn");
  const feedbackInaccurateBtn = document.getElementById("feedbackInaccurateBtn");
  const feedbackHelpfulBtn = document.getElementById("feedbackHelpfulBtn");
  const openDetailedFeedbackLink = document.getElementById("openDetailedFeedbackLink");
  const feedbackActionsGroup = document.getElementById("feedbackActionsGroup");
  const feedbackConfirmationBadge = document.getElementById("feedbackConfirmationBadge");
  const feedbackConfirmationText = document.getElementById("feedbackConfirmationText");
  const feedbackCorrectionDrawer = document.getElementById("feedbackCorrectionDrawer");
  const feedbackCorrectionSelect = document.getElementById("feedbackCorrectionSelect");
  const feedbackNotesInput = document.getElementById("feedbackNotesInput");
  const submitCorrectionBtn = document.getElementById("submitCorrectionBtn");

  // Feedback Modal Form Elements
  const openFeedbackModalBtn = document.getElementById("openFeedbackModalBtn");
  const feedbackModal = document.getElementById("feedbackModal");
  const closeFeedbackModalBtn = document.getElementById("closeFeedbackModalBtn");
  const cancelFeedbackModalBtn = document.getElementById("cancelFeedbackModalBtn");
  const dedicatedFeedbackForm = document.getElementById("dedicatedFeedbackForm");
  const modalStatementContext = document.getElementById("modalStatementContext");
  const modalCorrectedCategory = document.getElementById("modalCorrectedCategory");
  const modalFeedbackNotes = document.getElementById("modalFeedbackNotes");
  const modalNotesCount = document.getElementById("modalNotesCount");

  // Language Switcher
  const langBtnHindi = document.getElementById("langBtnHindi");
  const langBtnEnglish = document.getElementById("langBtnEnglish");

  // Voice AI Controls & State
  const autoVoiceToggle = document.getElementById("autoVoiceToggle");
  const voiceInputBtn = document.getElementById("voiceInputBtn");
  const voiceMicStatus = document.getElementById("voiceMicStatus");
  const playVoiceBtn = document.getElementById("playVoiceBtn");
  const voicePlayIcon = document.getElementById("voicePlayIcon");
  const replayVoiceBtn = document.getElementById("replayVoiceBtn");
  const stopVoiceBtn = document.getElementById("stopVoiceBtn");
  const voiceStatusSub = document.getElementById("voiceStatusSub");
  const audioEqualizer = document.getElementById("audioEqualizer");

  // KPI Ribbon
  const kpiTotalAssessments = document.getElementById("kpiTotalAssessments");
  const kpiAvgConfidence = document.getElementById("kpiAvgConfidence");
  const kpiFeedbackCount = document.getElementById("kpiFeedbackCount");
  const activeModelLabel = document.getElementById("activeModelLabel");

  // Visualizer Outcome Elements
  const emptyState = document.getElementById("emptyState");
  const resultsContainer = document.getElementById("resultsContainer");
  const predictedLabel = document.getElementById("predictedLabel");
  const outcomeDescription = document.getElementById("outcomeDescription");
  const confidenceValue = document.getElementById("confidenceValue");
  const confidenceRingFill = document.getElementById("confidenceRingFill");
  const riskLevelBadge = document.getElementById("riskLevelBadge");
  const riskLevelText = document.getElementById("riskLevelText");
  const detectedIndicatorsRow = document.getElementById("detectedIndicatorsRow");
  const downloadReportBtn = document.getElementById("downloadReportBtn");
  const tokenCloud = document.getElementById("tokenCloud");
  const featureWeightsList = document.getElementById("featureWeightsList");
  const probabilitiesGrid = document.getElementById("probabilitiesGrid");
  const solutionsListContainer = document.getElementById("solutionsListContainer");
  const solutionsBlockTitle = document.getElementById("solutionsBlockTitle");
  const solutionsBlockSub = document.getElementById("solutionsBlockSub");

  // Bottom Tabs Elements
  const navTabBtns = document.querySelectorAll(".nav-tab-btn");
  const tabPanels = document.querySelectorAll(".tab-panel");
  const historyBadgeCount = document.getElementById("historyBadgeCount");

  // Analytics Tab Elements
  const analyticsCategoryBars = document.getElementById("analyticsCategoryBars");
  const dbTotalCount = document.getElementById("dbTotalCount");
  const dbMeanConf = document.getElementById("dbMeanConf");
  const dbFeedbackCount = document.getElementById("dbFeedbackCount");

  // History Tab Elements
  const historyListContainer = document.getElementById("historyListContainer");
  const historySearchInput = document.getElementById("historySearchInput");
  const historyCategoryFilter = document.getElementById("historyCategoryFilter");
  const clearHistoryBtn = document.getElementById("clearHistoryBtn");

  // Architecture Specs Elements
  const specModelName = document.getElementById("specModelName");
  const specVectorization = document.getElementById("specVectorization");
  const specNgramRange = document.getElementById("specNgramRange");
  const specClassesCount = document.getElementById("specClassesCount");
  const globalClassesGrid = document.getElementById("globalClassesGrid");

  // In-memory state
  let loadedPresets = [];
  let currentPredictionData = null;
  let isSpeechRecording = false;
  let speechRecognizer = null;
  let currentSpeechUtterance = null;
  let isVoiceSpeaking = false;

  // Condition Descriptions Dictionary
  const conditionDescriptions = {
    "Normal": "Linguistic markers reflect balanced emotional state, resilience, optimism, and absence of acute psychological distress.",
    "Depression": "Linguistic patterns indicate persistent melancholy, emotional exhaustion, loss of interest, and feelings of emptiness.",
    "Suicidal": "High clinical urgency detected: Expressions reflect deep hopelessness, perceived burdensomeness, and crisis markers.",
    "Anxiety": "Patterns indicate hyper-vigilance, racing panic sensations, trembling dread, and anticipatory anxiety.",
    "Bipolar": "Linguistic markers reflect cyclical shifts between manic grandiosity / euphoria and severe depressive crashes.",
    "Stress": "Indicators suggest acute burnout, cognitive overload, deadline anxiety, and chronic exhaustion.",
    "Personality disorder": "Linguistic patterns indicate emotional dysregulation, identity distress, or profound fear of abandonment."
  };

  const conditionDescriptionsHindi = {
    "Normal": "भाषा में भावनात्मक संतुलन, सकारात्मक सोच, मानसिक शांति और किसी प्रकार के गंभीर तनाव की अनुपस्थिति दिखाई देती है।",
    "Depression": "भाषा में गहरी उदासी, अकेलापन, काम में मन न लगना और मानसिक खालीपन के स्पष्ट संकेत मिले हैं।",
    "Suicidal": "अत्यधिक गंभीर स्थिति: भाषा में गहरी निराशा और संकट के संकेत हैं। कृपया तुरंत सहायता प्राप्त करें।",
    "Anxiety": "भाषा में घबराहट, अत्यधिक चिंता, अनहोनी का डर और बेचैनी के स्पष्ट लक्षण पाए गए हैं।",
    "Bipolar": "भाषा में मनोदशा में अत्यधिक उतार-चढ़ाव (अत्यधिक उत्तेजना या अचानक अवसाद) के संकेत मिले हैं।",
    "Stress": "भाषा में अत्यधिक मानसिक दबाव, काम का बोझ, थकान और मानसिक तनाव के लक्षण हैं।",
    "Personality disorder": "भाषा में तीव्र भावनात्मक अस्थिरता और मन में लगातार उथल-पुथल के संकेत हैं।"
  };

  const HINDI_CONDITION_NAMES = {
    "Normal": "सामान्य / स्वस्थ (Normal)",
    "Depression": "अवसाद / डिप्रेशन (Depression)",
    "Suicidal": "अत्यधिक संकट (Suicidal)",
    "Anxiety": "चिंता व घबराहट (Anxiety)",
    "Bipolar": "द्विध्रुवी विकार (Bipolar)",
    "Stress": "मानसिक तनाव (Stress)",
    "Personality disorder": "व्यक्तित्व विकार (Personality Disorder)"
  };

  // Condition-Specific Evidence-Based Solutions & Coping Strategies
  const CONDITION_SOLUTIONS_EN = {
    "Depression": [
      { icon: "fa-person-walking", title: "Behavioral Activation", desc: "Take a gentle 5-minute walk outside in natural sunlight to boost dopamine and serotonin levels." },
      { icon: "fa-list-check", title: "Micro-Task Strategy", desc: "Break overwhelming routines into a single 2-minute actionable step without pressuring yourself." },
      { icon: "fa-handshake-angle", title: "Supportive Connection", desc: "Send a quick check-in message to a trusted friend, family member, or professional counselor." }
    ],
    "Anxiety": [
      { icon: "fa-wind", title: "4-7-8 Breathing Technique", desc: "Inhale gently for 4s, hold breath for 7s, and exhale slowly for 8s to calm the autonomic nervous system." },
      { icon: "fa-eye", title: "5-4-3-2-1 Grounding Method", desc: "Acknowledge 5 things you can see, 4 you can touch, 3 you hear, 2 you smell, and 1 you taste." },
      { icon: "fa-mug-hot", title: "Stimulant Reduction", desc: "Limit caffeine or energy drinks and replace them with warm water or soothing chamomile tea." }
    ],
    "Stress": [
      { icon: "fa-hourglass-half", title: "Pomodoro & Mindful Breaks", desc: "Work in focused 25-minute intervals followed by 5 minutes of deliberate screen-free rest." },
      { icon: "fa-shield-halved", title: "Cognitive Boundary Setting", desc: "Politely decline non-essential demands to protect and replenish your mental bandwidth." },
      { icon: "fa-spa", title: "Progressive Muscle Relaxation", desc: "Tense each muscle group for 5 seconds and slowly release tension from head to toe." }
    ],
    "Suicidal": [
      { icon: "fa-phone-volume", title: "Immediate 24/7 Helpline", desc: "Call or text 988 (USA) or 14416 / 9152987821 (Tele-MANAS India) for compassionate, free support." },
      { icon: "fa-house-chimney-user", title: "Safe Social Environment", desc: "Reach out to a trusted loved one and remain in a supportive, connected environment." },
      { icon: "fa-heart-pulse", title: "One Hour at a Time", desc: "Focus strictly on getting through the current hour safely. This acute pain is temporary." }
    ],
    "Bipolar": [
      { icon: "fa-moon", title: "Circadian Rhythm Stabilization", desc: "Maintain strict, consistent sleep-wake cycles every day to regulate neurotransmitter balance." },
      { icon: "fa-chart-line", title: "Mood & Energy Journaling", desc: "Track daily sleep hours and emotional intensity to spot early triggers before mood episodes." },
      { icon: "fa-user-doctor", title: "Clinical Support Adherence", desc: "Maintain regular consultations with your prescribing psychiatrist or clinical therapist." }
    ],
    "Personality disorder": [
      { icon: "fa-snowflake", title: "TIPP Distress Tolerance", desc: "Splash cold water on your face or hold an ice cube to rapidly de-escalate emotional intensity." },
      { icon: "fa-brain", title: "Mindful Defusion", desc: "Observe intense emotions non-judgmentally as temporary weather passing through the mind." },
      { icon: "fa-comments", title: "Assertive Communication", desc: "Express emotional needs clearly and calmly while maintaining healthy boundaries." }
    ],
    "Normal": [
      { icon: "fa-dumbbell", title: "Sustain Healthy Habits", desc: "Keep up regular physical activity, balanced nutrition, and restorative 7-8 hours of sleep." },
      { icon: "fa-book-bookmark", title: "Daily Gratitude Practice", desc: "Write down 3 positive moments or personal accomplishments from your day." },
      { icon: "fa-seedling", title: "Mindful Presence", desc: "Dedicate 5-10 minutes each day to meditation, deep reflection, or hobbies you enjoy." }
    ]
  };

  const CONDITION_SOLUTIONS_HI = {
    "Depression": [
      { icon: "fa-person-walking", title: "हल्की सैर व धूप (Behavioral Activation)", desc: "रोजाना 5-10 मिनट ताजी हवा और धूप में हल्की सैर करें, जिससे मन को ऊर्जा और शांति मिले।" },
      { icon: "fa-list-check", title: "छोटे लक्ष्य बनाएं (Micro-Tasks)", desc: "बड़े कामों के दबाव से बचें और सिर्फ एक छोटे से काम से शुरुआत करें।" },
      { icon: "fa-handshake-angle", title: "अपनों से बात करें (Social Connection)", desc: "अपने किसी करीबी दोस्त या परिवार के सदस्य से अपनी भावनाएं साझा करें।" }
    ],
    "Anxiety": [
      { icon: "fa-wind", title: "4-7-8 गहरी सांस लें (Deep Breathing)", desc: "4 सेकंड सांस अंदर लें, 7 सेकंड रोकें, और 8 सेकंड में धीरे-धीरे छोड़ें।" },
      { icon: "fa-eye", title: "5-4-3-2-1 ग्राउंडिंग तकनीक", desc: "अपने आस-पास की 5 चीजों को देखें, 4 को छुएं, 3 आवाजों को सुनें और मन को शांत करें।" },
      { icon: "fa-mug-hot", title: "कैफीन कम करें", desc: "चाय या कॉफी की जगह गुनगुना पानी या हर्बल टी पिएं।" }
    ],
    "Stress": [
      { icon: "fa-hourglass-half", title: "काम के बीच ब्रेक लें (Pomodoro)", desc: "हर 25 मिनट काम के बाद 5 मिनट का शांत विश्राम अवश्य लें।" },
      { icon: "fa-shield-halved", title: "सीमाएं तय करें (Set Boundaries)", desc: "मानसिक शांति बनाए रखने के लिए अनावश्यक जिम्मेदारियों से बचें।" },
      { icon: "fa-spa", title: "मांसपेशियों को आराम दें", desc: "सिर से पैर तक की मांसपेशियों को धीरे-धीरे ढीला छोड़ें और तनाव मुक्त महसूस करें।" }
    ],
    "Suicidal": [
      { icon: "fa-phone-volume", title: "तत्काल हेल्पलाइन से संपर्क करें", desc: "मुफ्त व गोपनीय सहायता के लिए तुरंत 14416 या 9152987821 (Tele-MANAS) पर कॉल करें।" },
      { icon: "fa-house-chimney-user", title: "अकेले न रहें (Safe Space)", desc: "अपने किसी विश्वस्त साथी के साथ रहें और सुरक्षित वातावरण बनाए रखें।" },
      { icon: "fa-heart-pulse", title: "धैर्य रखें", desc: "केवल अगले एक घंटे पर ध्यान दें। यह कठिन समय जरूर बीत जाएगा।" }
    ],
    "Bipolar": [
      { icon: "fa-moon", title: "नियमित नींद का समय (Sleep Rhythm)", desc: "रोजाना एक निश्चित समय पर सोएं और जागें।" },
      { icon: "fa-chart-line", title: "मूड डायरी बनाएं (Mood Tracking)", desc: "अपनी मनोदशा और ऊर्जा स्तर को ट्रैक करें ताकि शुरुआती लक्षणों को पहचाना जा सके।" },
      { icon: "fa-user-doctor", title: "चिकित्सक से परामर्श लें", desc: "अपने डॉक्टर या मनोचिकित्सक से नियमित संपर्क बनाए रखें।" }
    ],
    "Personality disorder": [
      { icon: "fa-snowflake", title: "ठंडे पानी का उपयोग (TIPP Skill)", desc: "तीव्र बेचैनी या घबराहट में चेहरे पर ठंडा पानी छिड़कें।" },
      { icon: "fa-brain", title: "विचारों को दूर से देखें (Mindfulness)", desc: "अपनी भावनाओं को बिना किसी फैसले के एक अस्थायी अनुभव समझें।" },
      { icon: "fa-comments", title: "शांत संवाद", desc: "अपनी बात को शांत और स्पष्ट तरीके से रखें।" }
    ],
    "Normal": [
      { icon: "fa-dumbbell", title: "सकारात्मक दिनचर्या बनाए रखें", desc: "नियमित व्यायाम, पौष्टिक भोजन और 7-8 घंटे की नींद की आदत जारी रखें।" },
      { icon: "fa-book-bookmark", title: "कृतज्ञता डायरी (Gratitude Journal)", desc: "आज की 3 अच्छी बातों या खुशियों को याद करके नोट करें।" },
      { icon: "fa-seedling", title: "दैनिक ध्यान व सुकून", desc: "मानसिक शांति और ताजगी बनाए रखने के लिए रोज 5-10 मिनट ध्यान करें।" }
    ]
  };

  function getConditionClass(category) {
    if (!category) return "Normal";
    if (category.startsWith("Personality")) return "Personality";
    return category;
  }

  function containsHindi(text) {
    if (!text) return false;
    if (/[\u0900-\u097F]/.test(text)) return true;
    const hinglishKeywords = /\b(mujhe|mera|meri|mere|bahut|bohot|udas|udasi|akela|akelapan|neend|ghabrahat|chinta|tanav|pareshaan|pareshan|dar|marne|khush|shant|lag|raha|rahi|nahi|aati|hoti|karna|karta|karte|hoon|hona|kuch|sab|kya|kyun|hota)\b/i;
    return hinglishKeywords.test(text);
  }

  // =========================================================================
  // 1. Language Toggle Logic
  // =========================================================================
  function setVoiceLanguage(lang) {
    voiceLang = lang;
    if (lang === "hi-IN") {
      if (langBtnHindi) langBtnHindi.classList.add("active");
      if (langBtnEnglish) langBtnEnglish.classList.remove("active");
      if (voiceMicStatus && !isSpeechRecording) voiceMicStatus.textContent = "हिंदी में बोलें";
    } else {
      if (langBtnEnglish) langBtnEnglish.classList.add("active");
      if (langBtnHindi) langBtnHindi.classList.remove("active");
      if (voiceMicStatus && !isSpeechRecording) voiceMicStatus.textContent = "Voice Input";
    }

    if (speechRecognizer) {
      speechRecognizer.lang = voiceLang;
    }
  }

  if (langBtnHindi) langBtnHindi.addEventListener("click", () => setVoiceLanguage("hi-IN"));
  if (langBtnEnglish) langBtnEnglish.addEventListener("click", () => setVoiceLanguage("en-US"));

  // =========================================================================
  // 2. Initial Dynamic Data Loading
  // =========================================================================
  loadPresetExamples();
  loadAnalyticsSummary();
  loadHistoryRecords();
  loadModelArchitectureInfo();
  loadModelsCatalog();
  initSpeechRecognition();
  initFeedbackControls();

  // Model Dropdown Change Handler
  if (modelSelectDropdown) {
    modelSelectDropdown.addEventListener("change", async () => {
      const chosen = modelSelectDropdown.value;
      const opt = modelSelectDropdown.options[modelSelectDropdown.selectedIndex];
      if (activeModelBadge) activeModelBadge.textContent = opt ? opt.text.split("•")[0].trim() : chosen;
      if (activeModelLabel) activeModelLabel.textContent = `${chosen.replace("_", " ").toUpperCase()} Active`;
      try {
        await fetch("/api/models/select", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model_name: chosen })
        });
      } catch (e) {
        console.warn("Model select error:", e);
      }
    });
  }

  // =========================================================================
  // 3. Tab Navigation Logic
  // =========================================================================
  navTabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-tab");
      navTabBtns.forEach(b => b.classList.remove("active"));
      tabPanels.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPanel = document.getElementById(targetId);
      if (targetPanel) targetPanel.classList.add("active");

      if (targetId === "tab-analytics") {
        loadAnalyticsSummary();
      } else if (targetId === "tab-history") {
        loadHistoryRecords();
      } else if (targetId === "tab-architecture") {
        loadModelArchitectureInfo();
      }
    });
  });

  // =========================================================================
  // 4. User Controls & Real-Time Counters
  // =========================================================================
  function updateCounters() {
    const val = statementInput.value;
    currentCount.textContent = val.length;
    const words = val.trim() ? val.trim().split(/\s+/).length : 0;
    if (wordCount) wordCount.textContent = words;
  }

  statementInput.addEventListener("input", updateCounters);

  statementInput.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      executeInference();
    }
  });

  clearBtn.addEventListener("click", () => {
    statementInput.value = "";
    updateCounters();
    stopVoiceover();
    statementInput.focus();
  });

  if (randomizeBtn) {
    randomizeBtn.addEventListener("click", () => {
      if (loadedPresets.length > 0) {
        const randomIndex = Math.floor(Math.random() * loadedPresets.length);
        const sample = loadedPresets[randomIndex];
        statementInput.value = sample.text;
        updateCounters();
        executeInference();
      }
    });
  }

  topNWordsRange.addEventListener("input", () => {
    topNWordsDisplay.textContent = `${topNWordsRange.value} Top Tokens`;
  });

  analyzeBtn.addEventListener("click", () => executeInference());

  // Download / Print Clinical Report Button
  if (downloadReportBtn) {
    downloadReportBtn.addEventListener("click", downloadClinicalReport);
  }

  // =========================================================================
  // 5. Dynamic Preset Scenarios Fetching (/api/examples)
  // =========================================================================
  async function loadPresetExamples() {
    try {
      const res = await fetch("/api/examples");
      if (!res.ok) throw new Error("Could not fetch preset scenarios");
      const data = await res.json();
      loadedPresets = data.examples || [];

      if (!presetChipsContainer) return;
      presetChipsContainer.innerHTML = "";

      loadedPresets.forEach(item => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "preset-chip-btn";
        btn.innerHTML = `<span class="chip-dot ${getConditionClass(item.category)}"></span><span>${item.label || item.category}</span>`;
        btn.title = `Click to evaluate: "${item.text.substring(0, 60)}..."`;
        btn.addEventListener("click", () => {
          statementInput.value = item.text;
          updateCounters();
          executeInference();
        });
        presetChipsContainer.appendChild(btn);
      });
    } catch (err) {
      console.warn("Failed to load preset scenarios:", err);
      if (presetChipsContainer) {
        presetChipsContainer.innerHTML = `<span class="text-muted">Preset examples unavailable.</span>`;
      }
    }
  }

  // =========================================================================
  // 6. Speech-to-Text: Multilingual Voice Input (Microphone)
  // =========================================================================
  function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      if (voiceInputBtn) {
        voiceInputBtn.title = "Voice Input not supported in this browser. Please use Chrome/Edge.";
        voiceInputBtn.style.opacity = "0.6";
      }
      return;
    }

    speechRecognizer = new SpeechRecognition();
    speechRecognizer.continuous = false;
    speechRecognizer.interimResults = true;
    speechRecognizer.lang = voiceLang;

    speechRecognizer.onstart = () => {
      isSpeechRecording = true;
      voiceInputBtn.classList.add("recording");
      voiceMicStatus.textContent = voiceLang === "hi-IN" ? "सुन रहा हूँ... बोलिए" : "Listening... Speak now";
    };

    speechRecognizer.onresult = (event) => {
      let finalTranscript = "";
      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        } else {
          statementInput.value = event.results[i][0].transcript;
          updateCounters();
        }
      }
      if (finalTranscript) {
        statementInput.value = finalTranscript;
        updateCounters();
      }
    };

    speechRecognizer.onerror = (event) => {
      console.warn("Speech recognition error:", event.error);
      resetMicUI();
      if (event.error === "not-allowed") {
        alert("Microphone permission was denied. Please allow microphone access in your browser settings.");
      }
    };

    speechRecognizer.onend = () => {
      resetMicUI();
      const text = statementInput.value.trim();
      if (text.length >= 3) {
        executeInference();
      }
    };

    if (voiceInputBtn) {
      voiceInputBtn.addEventListener("click", () => {
        if (isSpeechRecording) {
          speechRecognizer.stop();
          resetMicUI();
        } else {
          try {
            stopVoiceover();
            speechRecognizer.lang = voiceLang;
            speechRecognizer.start();
          } catch (e) {
            console.warn("Speech recognition start failed:", e);
          }
        }
      });
    }
  }

  function resetMicUI() {
    isSpeechRecording = false;
    if (voiceInputBtn) voiceInputBtn.classList.remove("recording");
    if (voiceMicStatus) voiceMicStatus.textContent = voiceLang === "hi-IN" ? "हिंदी में बोलें" : "Voice Input";
  }

  // =========================================================================
  // 7. Text-to-Speech: Child AI Companion Voiceover (with Fillers, Giggles & Teaching)
  // =========================================================================
  function buildVoiceoverSummary(data) {
    const cat = data.predicted_category;
    const conf = (data.confidence * 100).toFixed(0);
    const topWords = (data.top_contributing_words || [])
      .slice(0, 3)
      .map(w => w.word)
      .join(", ");

    const isHindi = containsHindi(data.text) || voiceLang === "hi-IN";

    if (isHindi) {
      const hindiCat = HINDI_CONDITION_NAMES[cat] || cat;
      let script = `अरे सुनो! हम्म... मैंने आपकी पूरी बात बहुत ध्यान से सुनी। (हीही... मुस्कुराइए!) `;
      script += `बिल्कुल मत घबराओ, सब ठीक हो जाएगा! मुझे लगता है आप थोड़ा ${hindiCat} महसूस कर रहे हो, लगभग ${conf} प्रतिशत पक्का है। `;
      if (topWords) {
        script += `पता है, आपके ${topWords} वाले शब्दों ने मुझे यह बताया। `;
      }
      const sol = (CONDITION_SOLUTIONS_HI[cat] || [])[0];
      if (sol) {
        script += `लेकिन चिंता की कोई बात नहीं, मैं सिखाता हूँ आपको क्या करना है! चलो उठो, ${sol.title} — ${sol.desc} सच में बहुत अच्छा और हल्का महसूस होगा! `;
      }
      if (cat === "Suicidal" || cat === "Depression") {
        script += "और हां, आप कभी अकेले नहीं हो, हम सब आपके साथ हैं! अगर ज्यादा परेशानी लगे तो 14416 पर कॉल कर सकते हो। अब जल्दी से एक प्यारी सी स्माइल दो!";
      } else {
        script += "मुस्कुराओ अब, मैं हमेशा आपका प्यारा दोस्त बनकर आपके साथ हूँ!";
      }
      return { script, lang: "hi-IN" };
    } else {
      let script = `Hey there! Umm... I listened to everything you shared with me. (giggles... hehe!) `;
      script += `Don't worry at all, everything is going to be completely fine! It looks like you're going through a bit of ${cat}, with about ${conf} percent certainty. `;
      if (topWords) {
        script += `You know, words like ${topWords} gave me a little clue. `;
      }
      const sol = (CONDITION_SOLUTIONS_EN[cat] || [])[0];
      if (sol) {
        script += `But hey, let me teach you a super helpful trick to feel lighter! Ready? ${sol.title} — ${sol.desc} It really works wonders! `;
      }
      if (cat === "Suicidal" || cat === "Depression") {
        script += "Please remember, you are so deeply loved and never alone. Caring people at 988 or 14416 are always here for you. Come on, give me a little smile now!";
      } else {
        script += "Cheer up now, I'm always right here as your caring little AI companion!";
      }
      return { script, lang: "en-US" };
    }
  }

  function playVoiceover(customText) {
    if (!('speechSynthesis' in window)) {
      alert("Text-to-Speech is not supported in your browser.");
      return;
    }

    window.speechSynthesis.cancel();

    let spokenText = "";
    let targetLang = voiceLang;

    if (customText) {
      spokenText = customText;
    } else if (currentPredictionData) {
      const summary = buildVoiceoverSummary(currentPredictionData);
      spokenText = summary.script;
      targetLang = summary.lang;
    } else {
      spokenText = voiceLang === "hi-IN" ? "अरे! अभी बात करने के लिए कोई वाक्य नहीं है। (hehe!)" : "Hey! No statement to talk about yet. (hehe!)";
    }

    currentSpeechUtterance = new SpeechSynthesisUtterance(spokenText);
    currentSpeechUtterance.lang = targetLang;
    currentSpeechUtterance.rate = 1.02;
    currentSpeechUtterance.pitch = 1.38;

    const voices = window.speechSynthesis.getVoices();
    let selectedVoice = null;

    if (targetLang === "hi-IN") {
      selectedVoice = voices.find(v => (
        v.lang === "hi-IN" ||
        v.lang.startsWith("hi") ||
        v.name.toLowerCase().includes("swara") ||
        v.name.toLowerCase().includes("heera") ||
        v.name.toLowerCase().includes("hindi")
      ));
    } else {
      selectedVoice = voices.find(v => (
        v.name.includes("Samantha") ||
        v.name.includes("Zira") ||
        v.name.includes("Google UK English Female") ||
        v.name.includes("Natural") ||
        v.lang === "en-US"
      ));
    }

    if (selectedVoice) currentSpeechUtterance.voice = selectedVoice;

    currentSpeechUtterance.onstart = () => {
      isVoiceSpeaking = true;
      if (voicePlayIcon) voicePlayIcon.className = "fa-solid fa-pause";
      if (audioEqualizer) audioEqualizer.classList.remove("hidden");
      if (voiceStatusSub) voiceStatusSub.textContent = targetLang === "hi-IN" ? "Child AI दोस्त आपको समाधान सिखा रहा है... (hehe!)" : "Child AI companion is cheerfully speaking...";
    };

    currentSpeechUtterance.onend = () => {
      stopVoiceoverUI();
    };

    currentSpeechUtterance.onerror = () => {
      stopVoiceoverUI();
    };

    window.speechSynthesis.speak(currentSpeechUtterance);
  }

  function stopVoiceover() {
    if ('speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
    stopVoiceoverUI();
  }

  function stopVoiceoverUI() {
    isVoiceSpeaking = false;
    if (voicePlayIcon) voicePlayIcon.className = "fa-solid fa-play";
    if (audioEqualizer) audioEqualizer.classList.add("hidden");
    if (voiceStatusSub) voiceStatusSub.textContent = "Click play to listen to empathetic audio summary";
  }

  if (playVoiceBtn) {
    playVoiceBtn.addEventListener("click", () => {
      if (isVoiceSpeaking) {
        stopVoiceover();
      } else {
        playVoiceover();
      }
    });
  }

  if (replayVoiceBtn) {
    replayVoiceBtn.addEventListener("click", () => {
      stopVoiceover();
      playVoiceover();
    });
  }

  if (stopVoiceBtn) {
    stopVoiceBtn.addEventListener("click", stopVoiceover);
  }

  // =========================================================================
  // 8. Run Text Inference & Feature Attribution (/predict)
  // =========================================================================
  async function executeInference() {
    const text = statementInput.value.trim();
    if (!text || text.length < 3) {
      alert("Please enter or speak a statement of at least 3 characters for clinical linguistic analysis.");
      statementInput.focus();
      return;
    }

    const topN = parseInt(topNWordsRange.value, 10) || 6;
    const selectedModel = modelSelectDropdown ? modelSelectDropdown.value : undefined;
    setLoadingState(true);

    try {
      const payload = {
        text: text,
        top_n_words: topN
      };
      if (selectedModel) {
        payload.model_name = selectedModel;
      }

      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Inference failed.");
      }

      const predictionData = await response.json();
      currentPredictionData = predictionData;
      renderInferenceResults(predictionData);

      // Auto-Voiceover if enabled
      if (autoVoiceToggle && autoVoiceToggle.checked) {
        playVoiceover();
      }

      // Refresh Analytics and History
      loadAnalyticsSummary();
      loadHistoryRecords();
    } catch (err) {
      console.error("Inference Error:", err);
      alert("Analysis failed: " + err.message);
    } finally {
      setLoadingState(false);
    }
  }

  function setLoadingState(isLoading) {
    analyzeBtn.disabled = isLoading;
    if (isLoading) {
      loadingSpinner.classList.remove("hidden");
    } else {
      loadingSpinner.classList.add("hidden");
    }
  }

  // =========================================================================
  // 9. Render Prediction Outcome, Risk Level, XAI & Actionable Solutions
  // =========================================================================
  function renderInferenceResults(data) {
    emptyState.classList.add("hidden");
    resultsContainer.classList.remove("hidden");

    // 1. Outcome Hero Card
    const cat = data.predicted_category;
    const isHindi = containsHindi(data.text) || voiceLang === "hi-IN";
    predictedLabel.textContent = isHindi ? (HINDI_CONDITION_NAMES[cat] || cat) : cat;
    outcomeDescription.textContent = isHindi ? (conditionDescriptionsHindi[cat] || conditionDescriptions[cat]) : (conditionDescriptions[cat] || "Classification complete.");
    
    // Confidence Ring Meter
    const confRatio = Math.max(0, Math.min(1, data.confidence));
    const confPercent = (confRatio * 100).toFixed(1);
    confidenceValue.textContent = `${confPercent}%`;

    if (confidenceRingFill) {
      const circumference = 2 * Math.PI * 32;
      confidenceRingFill.style.strokeDasharray = `${circumference} ${circumference}`;
      const offset = circumference - (confRatio * circumference);
      confidenceRingFill.style.strokeDashoffset = offset;

      if (confRatio >= 0.8) {
        confidenceRingFill.style.stroke = "#10b981";
      } else if (confRatio >= 0.6) {
        confidenceRingFill.style.stroke = "#f59e0b";
      } else {
        confidenceRingFill.style.stroke = "#f43f5e";
      }
    }

    // 2. Risk Level Tier Calculation
    renderRiskLevel(cat, confRatio, isHindi);

    // 3. Detected Linguistic Markers
    renderDetectedMarkers(cat, data.top_contributing_words, isHindi);

    // 4. Word-Level Explainability (Token Cloud)
    renderTokenCloud(data.text, data.top_contributing_words);

    // 5. Feature Weights Ranking
    renderFeatureRankingBars(data.top_contributing_words);

    // 6. Multi-Class Probability Distribution
    renderProbabilityDistribution(data.probabilities, cat);

    // 7. Actionable Coping Solutions & Guidance
    renderSolutions(cat, isHindi);

    // 8. Reset Feedback UI for Current Assessment
    resetFeedbackUI();
  }

  function resetFeedbackUI() {
    if (feedbackConfirmationBadge) feedbackConfirmationBadge.classList.add("hidden");
    if (feedbackActionsGroup) feedbackActionsGroup.classList.remove("hidden");
    if (feedbackCorrectionDrawer) feedbackCorrectionDrawer.classList.add("hidden");
    if (feedbackNotesInput) feedbackNotesInput.value = "";
    if (feedbackCorrectionSelect) feedbackCorrectionSelect.value = "";
    document.querySelectorAll(".btn-feedback-action").forEach(b => b.classList.remove("active"));
  }

  function initFeedbackControls() {
    if (feedbackAccurateBtn) {
      feedbackAccurateBtn.addEventListener("click", async () => {
        if (!currentPredictionData || !currentPredictionData.id) return;
        feedbackAccurateBtn.classList.add("active");
        if (feedbackInaccurateBtn) feedbackInaccurateBtn.classList.remove("active");
        if (feedbackHelpfulBtn) feedbackHelpfulBtn.classList.remove("active");
        if (feedbackCorrectionDrawer) feedbackCorrectionDrawer.classList.add("hidden");

        await submitFeedback(currentPredictionData.id, "accurate");
        showFeedbackConfirmation("✅ Marked as Accurate! Thank you for validating.");
      });
    }

    if (feedbackInaccurateBtn) {
      feedbackInaccurateBtn.addEventListener("click", () => {
        if (!currentPredictionData || !currentPredictionData.id) return;
        feedbackInaccurateBtn.classList.add("active");
        if (feedbackAccurateBtn) feedbackAccurateBtn.classList.remove("active");
        if (feedbackHelpfulBtn) feedbackHelpfulBtn.classList.remove("active");
        if (feedbackCorrectionDrawer) feedbackCorrectionDrawer.classList.toggle("hidden");
      });
    }

    if (feedbackHelpfulBtn) {
      feedbackHelpfulBtn.addEventListener("click", async () => {
        if (!currentPredictionData || !currentPredictionData.id) return;
        feedbackHelpfulBtn.classList.add("active");
        if (feedbackAccurateBtn) feedbackAccurateBtn.classList.remove("active");
        if (feedbackInaccurateBtn) feedbackInaccurateBtn.classList.remove("active");
        if (feedbackCorrectionDrawer) feedbackCorrectionDrawer.classList.add("hidden");

        await submitFeedback(currentPredictionData.id, "helpful");
        showFeedbackConfirmation("💡 Marked as Helpful! Thank you.");
      });
    }

    if (submitCorrectionBtn) {
      submitCorrectionBtn.addEventListener("click", async () => {
        if (!currentPredictionData || !currentPredictionData.id) return;
        const correctedCat = feedbackCorrectionSelect ? feedbackCorrectionSelect.value : null;
        const notes = feedbackNotesInput ? feedbackNotesInput.value.trim() : null;

        await submitFeedback(currentPredictionData.id, "inaccurate", notes, correctedCat);
        if (feedbackCorrectionDrawer) feedbackCorrectionDrawer.classList.add("hidden");
        showFeedbackConfirmation("✅ Correction Saved! Thank you for improving our model.");
      });
    }

    // Modal Triggers
    if (openFeedbackModalBtn) {
      openFeedbackModalBtn.addEventListener("click", openFeedbackModal);
    }
    if (openDetailedFeedbackLink) {
      openDetailedFeedbackLink.addEventListener("click", openFeedbackModal);
    }
    if (closeFeedbackModalBtn) {
      closeFeedbackModalBtn.addEventListener("click", closeFeedbackModal);
    }
    if (cancelFeedbackModalBtn) {
      cancelFeedbackModalBtn.addEventListener("click", closeFeedbackModal);
    }
    if (feedbackModal) {
      feedbackModal.addEventListener("click", (e) => {
        if (e.target === feedbackModal) closeFeedbackModal();
      });
    }

    // Live Note Character Counter
    if (modalFeedbackNotes && modalNotesCount) {
      modalFeedbackNotes.addEventListener("input", () => {
        modalNotesCount.textContent = modalFeedbackNotes.value.length;
      });
    }

    // Dedicated Form Submission
    if (dedicatedFeedbackForm) {
      dedicatedFeedbackForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const selectedRating = document.querySelector('input[name="modalUserFeedback"]:checked')?.value || "accurate";
        const correctedCat = modalCorrectedCategory ? modalCorrectedCategory.value : null;
        const notes = modalFeedbackNotes ? modalFeedbackNotes.value.trim() : "";
        const targetId = currentPredictionData ? currentPredictionData.id : null;

        try {
          const payload = {
            user_feedback: selectedRating,
            feedback_notes: notes || null,
            corrected_category: correctedCat || null
          };

          const endpoint = targetId ? `/api/history/${targetId}/feedback` : `/api/feedback`;
          const res = await fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
          });

          if (res.ok) {
            closeFeedbackModal();
            showFeedbackConfirmation("🎉 Thank you! Your feedback form has been recorded for model improvement.");
            loadAnalyticsSummary();
          } else {
            const err = await res.json();
            alert("Could not submit feedback: " + (err.detail || "Server error"));
          }
        } catch (err) {
          console.error("Feedback form error:", err);
          alert("Submission error: " + err.message);
        }
      });
    }
  }

  function openFeedbackModal() {
    if (!feedbackModal) return;
    if (modalStatementContext) {
      if (currentPredictionData && currentPredictionData.text) {
        modalStatementContext.innerHTML = `<strong>Statement:</strong> &ldquo;${currentPredictionData.text}&rdquo; <br><span class="text-muted">Classified as: <strong>${currentPredictionData.predicted_category}</strong> (${(currentPredictionData.confidence * 100).toFixed(1)}% certainty)</span>`;
      } else {
        modalStatementContext.innerHTML = `<em>No recent statement analyzed in current session. Submitting general AI model feedback.</em>`;
      }
    }
    feedbackModal.classList.remove("hidden");
    if (modalFeedbackNotes) modalFeedbackNotes.focus();
  }

  function closeFeedbackModal() {
    if (!feedbackModal) return;
    feedbackModal.classList.add("hidden");
    if (modalFeedbackNotes) {
      modalFeedbackNotes.value = "";
      if (modalNotesCount) modalNotesCount.textContent = "0";
    }
    if (modalCorrectedCategory) modalCorrectedCategory.value = "";
  }

  function showFeedbackConfirmation(msg) {
    if (feedbackConfirmationText) feedbackConfirmationText.textContent = msg;
    if (feedbackConfirmationBadge) feedbackConfirmationBadge.classList.remove("hidden");
    if (feedbackActionsGroup) feedbackActionsGroup.classList.add("hidden");
  }

  function renderRiskLevel(category, confidence, isHindi) {
    if (!riskLevelBadge || !riskLevelText) return;

    riskLevelBadge.className = "risk-level-badge";
    if (category === "Suicidal") {
      riskLevelBadge.classList.add("risk-crisis");
      riskLevelText.textContent = isHindi ? "अत्यधिक संकट (Critical Crisis)" : "Critical / Acute Crisis";
    } else if (category === "Depression" || category === "Bipolar") {
      riskLevelBadge.classList.add("risk-high");
      riskLevelText.textContent = isHindi ? "उच्च स्तर (High Clinical Risk)" : "High Clinical Risk";
    } else if (category === "Anxiety" || category === "Stress" || category === "Personality disorder") {
      riskLevelBadge.classList.add("risk-moderate");
      riskLevelText.textContent = isHindi ? "मध्यम स्तर (Moderate Risk)" : "Moderate Risk";
    } else {
      riskLevelBadge.classList.add("risk-low");
      riskLevelText.textContent = isHindi ? "सामान्य / सुरक्षित (Low Risk)" : "Low Risk / Healthy";
    }
  }

  function renderDetectedMarkers(category, topTokens, isHindi) {
    if (!detectedIndicatorsRow) return;
    detectedIndicatorsRow.innerHTML = "";

    const defaultMarkers = {
      "Depression": isHindi ? ["भावनात्मक उदासी", "अकेलेपन के संकेत", "ऊर्जा की कमी"] : ["Emotional Melancholy", "Social Isolation Cues", "Fatigue Language"],
      "Anxiety": isHindi ? ["घबराहट व डर", "अनहोनी की चिंता", "शारीरिक बेचैनी"] : ["Panic Sensations", "Anticipatory Dread", "Autonomic Overdrive"],
      "Stress": isHindi ? ["मानसिक दबाव", "काम का बोझ", "थकान व तनाव"] : ["Cognitive Overload", "Burnout Cues", "Performance Stress"],
      "Suicidal": isHindi ? ["अत्यधिक संकट", "गहरी निराशा", "मदद की आवश्यकता"] : ["Acute Hopelessness", "Burdensomeness", "Urgent Intervention"],
      "Bipolar": isHindi ? ["मनोदशा का उतार-चढ़ाव", "नींद में बदलाव"] : ["Affective Shifts", "Altered Sleep Rhythm"],
      "Personality disorder": isHindi ? ["भावनात्मक अस्थिरता", "मन में उथल-पुथल"] : ["Emotional Dysregulation", "Identity Distress"],
      "Normal": isHindi ? ["संतुलित सोच", "सकारात्मक भाव"] : ["Resilient Thinking", "Emotional Balance"]
    };

    const markers = defaultMarkers[category] || (isHindi ? ["मानसिक संकेत"] : ["Linguistic Markers"]);

    markers.forEach(m => {
      const span = document.createElement("span");
      span.className = "detected-badge-item";
      span.innerHTML = `<i class="fa-solid fa-circle-dot text-amber"></i> ${m}`;
      detectedIndicatorsRow.appendChild(span);
    });

    if (topTokens && topTokens.length > 0) {
      const topWord = topTokens[0].word;
      const span = document.createElement("span");
      span.className = "detected-badge-item";
      span.innerHTML = `<i class="fa-solid fa-tag text-terracotta"></i> "${topWord}"`;
      detectedIndicatorsRow.appendChild(span);
    }
  }

  function renderSolutions(category, isHindi) {
    if (!solutionsListContainer) return;
    solutionsListContainer.innerHTML = "";

    const solutions = (isHindi ? CONDITION_SOLUTIONS_HI[category] : CONDITION_SOLUTIONS_EN[category]) || (isHindi ? CONDITION_SOLUTIONS_HI["Normal"] : CONDITION_SOLUTIONS_EN["Normal"]);

    if (solutionsBlockTitle) {
      solutionsBlockTitle.textContent = isHindi ? `समाधान व व्यावहारिक कदम: ${HINDI_CONDITION_NAMES[category] || category}` : `Evidence-Based Coping Solutions for ${category}`;
    }
    if (solutionsBlockSub) {
      solutionsBlockSub.textContent = isHindi ? `इस स्थिति के लिए वैज्ञानिक रूप से प्रभावी कदम:` : `Tailored, evidence-based coping strategies and practical steps recommended for this condition:`;
    }

    solutions.forEach(sol => {
      const card = document.createElement("div");
      card.className = "solution-item-card";

      const iconBox = document.createElement("div");
      iconBox.className = "sol-icon-box";
      iconBox.innerHTML = `<i class="fa-solid ${sol.icon}"></i>`;

      const meta = document.createElement("div");
      meta.className = "sol-meta";

      const title = document.createElement("h5");
      title.className = "sol-title";
      title.textContent = sol.title;

      const desc = document.createElement("p");
      desc.className = "sol-desc";
      desc.textContent = sol.desc;

      meta.appendChild(title);
      meta.appendChild(desc);

      card.appendChild(iconBox);
      card.appendChild(meta);
      solutionsListContainer.appendChild(card);
    });
  }

  function renderTokenCloud(fullText, contributingWords) {
    tokenCloud.innerHTML = "";
    const wordScoreMap = {};
    (contributingWords || []).forEach(item => {
      wordScoreMap[item.word.toLowerCase()] = item.score;
    });

    const tokens = fullText.split(/(\s+)/);
    tokens.forEach(tok => {
      if (/^\s+$/.test(tok)) {
        tokenCloud.appendChild(document.createTextNode(" "));
        return;
      }

      const cleanWord = tok.toLowerCase().replace(/[^a-zA-Z\u0900-\u097F0-9_-]/g, "");
      const score = wordScoreMap[cleanWord];

      const tag = document.createElement("span");
      tag.className = "token-pill-tag";
      tag.textContent = tok;

      if (score !== undefined) {
        if (score >= 0) {
          tag.classList.add("weight-positive");
          tag.title = `Positive Weight: +${score.toFixed(4)}`;
        } else {
          tag.classList.add("weight-negative");
          tag.title = `Opposing Weight: ${score.toFixed(4)}`;
        }

        const scoreBadge = document.createElement("span");
        scoreBadge.className = "token-score-tag";
        scoreBadge.textContent = `(${score >= 0 ? "+" : ""}${score.toFixed(2)})`;
        tag.appendChild(scoreBadge);
      }

      tokenCloud.appendChild(tag);
    });
  }

  function renderFeatureRankingBars(features) {
    featureWeightsList.innerHTML = "";
    if (!features || features.length === 0) {
      featureWeightsList.innerHTML = "<p class='loading-state-text'>No distinct token weights extracted.</p>";
      return;
    }

    const maxAbsScore = Math.max(...features.map(f => Math.abs(f.score)), 0.001);

    features.forEach((f, idx) => {
      const row = document.createElement("div");
      row.className = "feature-ranking-row";

      const rankBadge = document.createElement("div");
      rankBadge.className = "feature-rank-badge";
      rankBadge.textContent = `#${idx + 1}`;

      const name = document.createElement("div");
      name.className = "feature-token-name";
      name.textContent = f.word;
      name.title = f.word;

      const track = document.createElement("div");
      track.className = "feature-track-line";

      const fill = document.createElement("div");
      fill.className = `feature-fill-bar ${f.score < 0 ? "negative" : ""}`;
      const widthPct = Math.min(100, Math.round((Math.abs(f.score) / maxAbsScore) * 100));
      fill.style.width = `${widthPct}%`;
      track.appendChild(fill);

      const scoreNum = document.createElement("div");
      scoreNum.className = "feature-score-num";
      scoreNum.textContent = `${f.score >= 0 ? "+" : ""}${f.score.toFixed(2)}`;

      row.appendChild(rankBadge);
      row.appendChild(name);
      row.appendChild(track);
      row.appendChild(scoreNum);
      featureWeightsList.appendChild(row);
    });
  }

  function renderProbabilityDistribution(probabilities, activeCategory) {
    probabilitiesGrid.innerHTML = "";
    if (!probabilities) return;

    const sortedEntries = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);

    sortedEntries.forEach(([catName, probVal]) => {
      const isWinner = (catName === activeCategory);
      const row = document.createElement("div");
      row.className = `prob-category-row ${isWinner ? "is-winner" : ""}`;

      const label = document.createElement("span");
      label.className = "prob-category-name";
      label.innerHTML = `<span class="condition-pill-badge ${getConditionClass(catName)}" style="padding: 2px 8px; font-size: 11px;">${catName}</span>`;

      const meterGroup = document.createElement("div");
      meterGroup.className = "prob-meter-group";

      const meterBg = document.createElement("div");
      meterBg.className = "prob-meter-bg";

      const meterFill = document.createElement("div");
      meterFill.className = "prob-meter-active-fill";
      const pct = (probVal * 100).toFixed(1);
      meterFill.style.width = `${pct}%`;
      meterBg.appendChild(meterFill);

      const pctVal = document.createElement("span");
      pctVal.className = "prob-percent-val";
      pctVal.textContent = `${pct}%`;

      meterGroup.appendChild(meterBg);
      meterGroup.appendChild(pctVal);

      row.appendChild(label);
      row.appendChild(meterGroup);
      probabilitiesGrid.appendChild(row);
    });
  }

  // =========================================================================
  // 10. Printable / PDF Clinical Report Generation
  // =========================================================================
  function downloadClinicalReport() {
    if (!currentPredictionData) {
      alert("Please execute an assessment first to generate a clinical summary report.");
      return;
    }

    const data = currentPredictionData;
    const cat = data.predicted_category;
    const conf = (data.confidence * 100).toFixed(1);
    const dateStr = new Date().toLocaleString();
    const solutions = CONDITION_SOLUTIONS_EN[cat] || CONDITION_SOLUTIONS_EN["Normal"];

    let tokenRows = "";
    (data.top_contributing_words || []).forEach((w, i) => {
      tokenRows += `<tr><td>#${i + 1}</td><td><strong>${w.word}</strong></td><td>${w.score >= 0 ? '+' : ''}${w.score.toFixed(4)}</td><td>${w.score >= 0 ? 'Diagnostic Positive' : 'Opposing'}</td></tr>`;
    });

    let solItems = "";
    solutions.forEach(s => {
      solItems += `<li><strong>${s.title}</strong>: ${s.desc}</li>`;
    });

    let probRows = "";
    Object.entries(data.probabilities || {}).sort((a, b) => b[1] - a[1]).forEach(([k, v]) => {
      probRows += `<tr><td>${k}</td><td><strong>${(v * 100).toFixed(2)}%</strong></td></tr>`;
    });

    const reportHtml = `
      <!DOCTYPE html>
      <html>
      <head>
        <title>MindLens XAI — Clinical Linguistic Assessment Report</title>
        <style>
          body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #1e293b; padding: 30px; margin: 0; line-height: 1.5; background: #fff; }
          .header { border-bottom: 2px solid #ea580c; padding-bottom: 12px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center; }
          .brand { font-size: 22px; font-weight: 800; color: #ea580c; }
          .meta-info { font-size: 12px; color: #64748b; text-align: right; }
          .section-title { font-size: 15px; font-weight: 700; color: #0f172a; margin-top: 20px; margin-bottom: 8px; border-bottom: 1px solid #e2e8f0; padding-bottom: 4px; }
          .statement-box { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 14px; font-style: italic; color: #334155; margin-bottom: 15px; }
          .kpi-row { display: flex; gap: 20px; margin-bottom: 15px; }
          .kpi-card { flex: 1; background: #fff7ed; border: 1px solid #fed7aa; border-radius: 6px; padding: 12px; }
          .kpi-label { font-size: 11px; text-transform: uppercase; color: #9a3412; font-weight: 700; }
          .kpi-val { font-size: 20px; font-weight: 800; color: #c2410c; }
          table { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 12.5px; }
          th, td { border: 1px solid #e2e8f0; padding: 7px 10px; text-align: left; }
          th { background: #f1f5f9; color: #475569; }
          ul { margin: 6px 0; padding-left: 20px; font-size: 13px; }
          li { margin-bottom: 6px; }
          .disclaimer { margin-top: 30px; font-size: 11px; color: #64748b; border-top: 1px dashed #cbd5e1; padding-top: 10px; }
          @media print { button { display: none; } }
        </style>
      </head>
      <body>
        <div class="header">
          <div>
            <div class="brand">MindLens XAI &bull; Clinical Assessment Summary</div>
            <div style="font-size: 12px; color: #475569;">Explainable AI Linguistic Risk Evaluation</div>
          </div>
          <div class="meta-info">
            <div><strong>Date:</strong> ${dateStr}</div>
            <div><strong>Architecture:</strong> TF-IDF Logistic Regression</div>
          </div>
        </div>

        <div class="section-title">Evaluated Statement</div>
        <div class="statement-box">"${data.text}"</div>

        <div class="kpi-row">
          <div class="kpi-card">
            <div class="kpi-label">Primary Indication</div>
            <div class="kpi-val">${cat}</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Model Certainty</div>
            <div class="kpi-val">${conf}%</div>
          </div>
          <div class="kpi-card">
            <div class="kpi-label">Risk Category</div>
            <div class="kpi-val">${cat === 'Suicidal' ? 'Critical Crisis' : (cat === 'Depression' || cat === 'Bipolar' ? 'High Risk' : (cat === 'Normal' ? 'Low Risk' : 'Moderate'))}</div>
          </div>
        </div>

        <div class="section-title">Word-Level Explainability (Log-Odds Attribution)</div>
        <table>
          <thead>
            <tr><th>Rank</th><th>Influential Token</th><th>Log-Odds Score</th><th>Clinical Weight Impact</th></tr>
          </thead>
          <tbody>
            ${tokenRows}
          </tbody>
        </table>

        <div class="section-title">Class Probability Breakdown (7 Taxonomies)</div>
        <table>
          <thead>
            <tr><th>Diagnostic Condition</th><th>Posterior Probability</th></tr>
          </thead>
          <tbody>
            ${probRows}
          </tbody>
        </table>

        <div class="section-title">Recommended Coping Steps &amp; Behavioral Solutions</div>
        <ul>
          ${solItems}
        </ul>

        <div class="disclaimer">
          <strong>Clinical Disclaimer:</strong> MindLens XAI is an empirical Natural Language Processing research tool and is <em>not an official diagnostic device</em>. If immediate crisis intervention is required, please reach out to <strong>Tele-MANAS (14416 / 1800-599-0019)</strong> in India or <strong>988 Suicide &amp; Crisis Lifeline</strong> in the USA/Canada.
        </div>
      </body>
      </html>
    `;

    const printWin = window.open('', '_blank', 'width=850,height=750');
    if (printWin) {
      printWin.document.write(reportHtml);
      printWin.document.close();
      printWin.focus();
      setTimeout(() => { printWin.print(); }, 400);
    } else {
      alert("Popup blocker prevented report opening. Please allow popups for this site.");
    }
  }

  // =========================================================================
  // 11. Dynamic SQLite Analytics Fetching (/api/analytics)
  // =========================================================================
  async function loadAnalyticsSummary() {
    try {
      const res = await fetch("/api/analytics");
      if (!res.ok) throw new Error("Could not fetch analytics");
      const data = await res.json();

      const total = data.total_predictions || 0;
      const avgConf = data.average_confidence ? (data.average_confidence * 100).toFixed(1) + "%" : "0.0%";
      const fbSummary = data.feedback_summary || {};
      const feedbackAccurate = fbSummary.accurate_count || data.feedback_distribution?.accurate || 0;
      const totalFb = fbSummary.total_feedback || 0;
      const accuracyRate = fbSummary.human_accuracy_pct_str || (totalFb > 0 ? ((feedbackAccurate / totalFb) * 100).toFixed(1) + "%" : "N/A");

      // Top Ribbon
      if (kpiTotalAssessments) kpiTotalAssessments.textContent = total.toLocaleString();
      if (kpiAvgConfidence) kpiAvgConfidence.textContent = avgConf;
      if (kpiFeedbackCount) kpiFeedbackCount.textContent = totalFb > 0 ? `${accuracyRate} (${feedbackAccurate}/${totalFb})` : `${feedbackAccurate} Verified`;

      // DB Summary
      if (dbTotalCount) dbTotalCount.textContent = `${total} Stored Inferences`;
      if (dbMeanConf) dbMeanConf.textContent = avgConf;
      if (dbFeedbackCount) dbFeedbackCount.textContent = `${totalFb} Total Ratings (${accuracyRate} Accurate)`;

      // Feedback Analytics Cards
      const fbAccurateCount = document.getElementById("fbAccurateCount");
      const fbAccuratePct = document.getElementById("fbAccuratePct");
      const fbInaccurateCount = document.getElementById("fbInaccurateCount");
      const fbInaccuratePct = document.getElementById("fbInaccuratePct");
      const fbCorrectionsCount = document.getElementById("fbCorrectionsCount");
      const feedbackAccuracyRateText = document.getElementById("feedbackAccuracyRateText");
      const feedbackCategoryList = document.getElementById("feedbackCategoryList");

      if (feedbackAccuracyRateText) {
        feedbackAccuracyRateText.textContent = totalFb > 0 ? `${accuracyRate} Human Accuracy` : "Awaiting User Feedback";
      }
      if (fbAccurateCount) fbAccurateCount.textContent = feedbackAccurate;
      if (fbAccuratePct) {
        fbAccuratePct.textContent = totalFb > 0 ? `${((feedbackAccurate / totalFb) * 100).toFixed(1)}% of total ratings` : "No ratings yet";
      }
      if (fbInaccurateCount) fbInaccurateCount.textContent = fbSummary.inaccurate_count || 0;
      if (fbInaccuratePct) {
        const inacc = fbSummary.inaccurate_count || 0;
        fbInaccuratePct.textContent = totalFb > 0 ? `${((inacc / totalFb) * 100).toFixed(1)}% of total ratings` : "No ratings yet";
      }
      if (fbCorrectionsCount) fbCorrectionsCount.textContent = fbSummary.corrections_count || 0;

      // Category-Specific Feedback List
      if (feedbackCategoryList) {
        const catFb = fbSummary.category_feedback || {};
        const catEntries = Object.entries(catFb);
        if (catEntries.length === 0) {
          feedbackCategoryList.innerHTML = `<p class="loading-state-text">No user feedback submitted yet. Rate predictions above to see real-time human accuracy trends.</p>`;
        } else {
          feedbackCategoryList.innerHTML = "";
          catEntries.forEach(([catName, counts]) => {
            const acc = counts.accurate || 0;
            const inacc = counts.inaccurate || 0;
            const helpful = counts.helpful || 0;
            const totalForCat = acc + inacc + helpful;
            const pct = totalForCat > 0 ? (((acc + helpful) / totalForCat) * 100).toFixed(0) : 0;

            const row = document.createElement("div");
            row.className = "fb-cat-row";
            row.innerHTML = `
              <span class="fb-cat-name">
                <span class="condition-pill-badge ${getConditionClass(catName)}" style="padding: 2px 7px; font-size: 11px; margin-right: 6px;">${catName}</span>
              </span>
              <span class="fb-cat-stat">
                <strong class="text-emerald">${pct}% Accurate</strong> (${acc + helpful}/${totalForCat})
              </span>
            `;
            feedbackCategoryList.appendChild(row);
          });
        }
      }

      // Category Breakdown Bars
      if (analyticsCategoryBars) {
        const catDist = data.category_distribution || {};
        const entries = Object.entries(catDist);

        if (entries.length === 0) {
          analyticsCategoryBars.innerHTML = "<p class='loading-state-text'>No assessments stored yet. Run an analysis to generate live metrics.</p>";
          return;
        }

        analyticsCategoryBars.innerHTML = "";
        const maxCount = Math.max(...entries.map(([_, v]) => v.count), 1);

        entries.forEach(([catName, stats]) => {
          const item = document.createElement("div");
          item.className = "dist-bar-item";

          const header = document.createElement("div");
          header.className = "dist-bar-header";

          const title = document.createElement("span");
          title.innerHTML = `<span class="condition-pill-badge ${getConditionClass(catName)}" style="padding: 2px 7px; font-size: 10.5px; margin-right: 6px;">${catName}</span> <strong>${stats.count}</strong>`;

          const avgBadge = document.createElement("span");
          avgBadge.className = "text-muted";
          avgBadge.textContent = `Avg: ${(stats.avg_confidence * 100).toFixed(1)}%`;

          header.appendChild(title);
          header.appendChild(avgBadge);

          const track = document.createElement("div");
          track.className = "dist-bar-track";

          const fill = document.createElement("div");
          fill.className = "dist-bar-fill";
          fill.style.width = `${Math.min(100, Math.round((stats.count / maxCount) * 100))}%`;
          track.appendChild(fill);

          item.appendChild(header);
          item.appendChild(track);
          analyticsCategoryBars.appendChild(item);
        });
      }
    } catch (err) {
      console.warn("Analytics error:", err);
    }
  }

  // =========================================================================
  // 12. Dynamic SQLite History Fetching & Actions (/api/history)
  // =========================================================================
  async function loadHistoryRecords() {
    if (!historyListContainer) return;
    const category = historyCategoryFilter ? historyCategoryFilter.value : "";
    const search = historySearchInput ? historySearchInput.value.trim() : "";

    let url = `/api/history?limit=40`;
    if (category) url += `&category=${encodeURIComponent(category)}`;
    if (search) url += `&search=${encodeURIComponent(search)}`;

    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error("Failed to load assessment history");
      const data = await res.json();

      const total = data.total || 0;
      if (historyBadgeCount) historyBadgeCount.textContent = total;

      if (!data.items || data.items.length === 0) {
        historyListContainer.innerHTML = "<p class='loading-state-text'>No assessment records found matching criteria.</p>";
        return;
      }

      historyListContainer.innerHTML = "";
      data.items.forEach(record => {
        const card = document.createElement("div");
        card.className = "history-card-item";
        card.id = `history-card-${record.id}`;

        const top = document.createElement("div");
        top.className = "history-item-top";

        const badge = document.createElement("span");
        badge.className = `condition-pill-badge ${getConditionClass(record.predicted_category)}`;
        badge.textContent = `${record.predicted_category} (${(record.confidence * 100).toFixed(1)}%)`;

        const time = document.createElement("span");
        time.className = "history-timestamp";
        const dateObj = new Date(record.created_at);
        time.textContent = isNaN(dateObj) ? record.created_at : dateObj.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', month: 'short', day: 'numeric' });

        top.appendChild(badge);
        top.appendChild(time);

        const statementText = document.createElement("p");
        statementText.className = "history-statement-body";
        statementText.textContent = record.statement_text;

        const footer = document.createElement("div");
        footer.className = "history-item-footer";

        const feedbackInfo = document.createElement("span");
        feedbackInfo.className = "feedback-status-text";
        if (record.user_feedback) {
          feedbackInfo.innerHTML = `<i class="fa-solid fa-circle-check text-terracotta"></i> Feedback: ${record.user_feedback}`;
        } else {
          feedbackInfo.textContent = `Model: ${record.model_name || "TF-IDF + ML"}`;
        }

        const actions = document.createElement("div");
        actions.className = "history-actions-row";

        // Inspect Button
        const inspectBtn = document.createElement("button");
        inspectBtn.className = "btn-card-action";
        inspectBtn.innerHTML = '<i class="fa-solid fa-magnifying-glass"></i> Inspect';
        inspectBtn.title = "Inspect and reload into workspace";
        inspectBtn.addEventListener("click", () => {
          statementInput.value = record.statement_text;
          updateCounters();
          currentPredictionData = {
            text: record.statement_text,
            predicted_category: record.predicted_category,
            confidence: record.confidence,
            probabilities: record.probabilities || {},
            top_contributing_words: record.top_contributing_words || []
          };
          renderInferenceResults(currentPredictionData);
          if (autoVoiceToggle && autoVoiceToggle.checked) {
            playVoiceover();
          }
          window.scrollTo({ top: 300, behavior: 'smooth' });
        });

        // Thumbs Up Feedback
        const thumbsUpBtn = document.createElement("button");
        thumbsUpBtn.className = "btn-card-action";
        thumbsUpBtn.innerHTML = '<i class="fa-solid fa-thumbs-up"></i>';
        thumbsUpBtn.title = "Mark as accurate";
        thumbsUpBtn.addEventListener("click", async () => {
          await submitFeedback(record.id, "accurate");
          feedbackInfo.innerHTML = '<i class="fa-solid fa-circle-check text-terracotta"></i> Accurate';
          loadAnalyticsSummary();
        });

        // Delete Button
        const delBtn = document.createElement("button");
        delBtn.className = "btn-card-action danger";
        delBtn.innerHTML = '<i class="fa-solid fa-trash-can"></i>';
        delBtn.title = "Delete record";
        delBtn.addEventListener("click", async () => {
          await deleteRecord(record.id);
        });

        actions.appendChild(inspectBtn);
        actions.appendChild(thumbsUpBtn);
        actions.appendChild(delBtn);

        footer.appendChild(feedbackInfo);
        footer.appendChild(actions);

        card.appendChild(top);
        card.appendChild(statementText);
        card.appendChild(footer);
        historyListContainer.appendChild(card);
      });
    } catch (err) {
      console.error("History error:", err);
      historyListContainer.innerHTML = `<p class='loading-state-text text-danger'>Failed to load history: ${err.message}</p>`;
    }
  }

  if (historyCategoryFilter) historyCategoryFilter.addEventListener("change", loadHistoryRecords);
  if (historySearchInput) {
    let debounceTimer;
    historySearchInput.addEventListener("input", () => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(loadHistoryRecords, 300);
    });
  }

  async function submitFeedback(id, val, notes = null, correctedCat = null) {
    if (!id) return;
    try {
      const payload = { user_feedback: val };
      if (notes) payload.feedback_notes = notes;
      if (correctedCat) payload.corrected_category = correctedCat;

      await fetch(`/api/history/${id}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      loadAnalyticsSummary();
    } catch (e) {
      console.warn("Feedback error:", e);
    }
  }

  async function deleteRecord(id) {
    if (!confirm("Are you sure you want to delete this assessment?")) return;
    try {
      const res = await fetch(`/api/history/${id}`, { method: "DELETE" });
      if (res.ok) {
        const el = document.getElementById(`history-card-${id}`);
        if (el) el.remove();
        loadAnalyticsSummary();
      }
    } catch (e) {
      console.warn("Delete error:", e);
    }
  }

  if (clearHistoryBtn) {
    clearHistoryBtn.addEventListener("click", async () => {
      if (!confirm("Are you sure you want to wipe all assessment history from SQLite?")) return;
      try {
        const res = await fetch("/api/history", { method: "DELETE" });
        if (res.ok) {
          loadHistoryRecords();
          loadAnalyticsSummary();
        }
      } catch (e) {
        console.warn("Clear history error:", e);
      }
    });
  }

  // =========================================================================
  // 13. Dynamic Model Architecture & Vocabulary Fetching (/api/info)
  // =========================================================================
  let isArchitectureLoaded = false;
  async function loadModelArchitectureInfo() {
    if (isArchitectureLoaded) return;
    try {
      const res = await fetch("/api/info");
      if (!res.ok) return;
      const info = await res.json();

      if (specModelName && info.model_name) {
        specModelName.textContent = info.model_name.replace("_", " ").toUpperCase();
      }
      if (activeModelLabel && info.model_name) {
        activeModelLabel.textContent = `${info.model_name.replace("_", " ").toUpperCase()} Active`;
      }
      if (specNgramRange && info.ngram_range && info.max_features) {
        specNgramRange.textContent = `N-Grams (${info.ngram_range.join(",")}), ${info.max_features.toLocaleString()} Max Features`;
      }
      if (specClassesCount && info.classes) {
        specClassesCount.textContent = `${info.classes.length} Diagnostic Classes`;
      }

      if (globalClassesGrid && info.top_class_features) {
        globalClassesGrid.innerHTML = "";
        Object.entries(info.top_class_features).forEach(([catName, keywords]) => {
          const box = document.createElement("div");
          box.className = "global-class-card";

          const title = document.createElement("div");
          title.className = "global-class-header";
          title.textContent = catName;

          const wrap = document.createElement("div");
          wrap.className = "global-keywords-wrap";

          (keywords || []).forEach(kw => {
            const pill = document.createElement("span");
            pill.className = "global-keyword-pill";
            pill.textContent = kw;
            wrap.appendChild(pill);
          });

          box.appendChild(title);
          box.appendChild(wrap);
          globalClassesGrid.appendChild(box);
        });
      }

      isArchitectureLoaded = true;
    } catch (err) {
      console.warn("Model info error:", err);
    }
  }

  // =========================================================================
  // 14. Dynamic Model Catalog & Benchmark Percentage Sync (/api/models)
  // =========================================================================
  async function loadModelsCatalog() {
    try {
      const res = await fetch("/api/models");
      if (!res.ok) return;
      const data = await res.json();
      const models = data.models || [];
      const best = data.best_model || models.find(m => m.is_best) || models[0];

      if (bestModelPill && best) {
        bestModelPill.innerHTML = `<i class="fa-solid fa-crown text-amber"></i> Best: <strong>${best.short_name || best.name} (${best.accuracy})</strong>`;
      }

      if (modelComparisonPills && models.length > 0) {
        modelComparisonPills.innerHTML = "";
        const colorClasses = ["text-emerald", "text-cyan", "text-amber", "text-purple"];
        const iconClasses = ["fa-star text-amber", "fa-shield-halved text-cyan", "fa-tree text-green", "fa-chart-simple text-purple"];

        models.forEach((m, idx) => {
          const card = document.createElement("div");
          card.className = `model-compare-pill ${m.is_best ? "best-model-highlight" : ""}`;
          card.id = `pill-${m.id}`;

          const icon = iconClasses[idx % iconClasses.length];
          const color = colorClasses[idx % colorClasses.length];

          card.innerHTML = `
            <div class="model-compare-head">
              <span class="model-icon"><i class="fa-solid ${icon}"></i></span>
              <span class="model-title">${m.short_name || m.name}</span>
              ${m.is_best ? '<span class="crown-tag">🏆 Best Model</span>' : ''}
            </div>
            <div class="model-compare-pct ${color}">${m.accuracy} Accuracy</div>
            <div class="model-compare-sub">F1-Score: ${m.f1_score} &bull; Latency: ${m.latency || "~4ms"}</div>
          `;
          modelComparisonPills.appendChild(card);
        });
      }
    } catch (err) {
      console.warn("Models catalog error:", err);
    }
  }
});
