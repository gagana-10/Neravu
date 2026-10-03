import streamlit as st
import requests

from api import send_message
from styles import apply_styles


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(page_title="Neravu", page_icon="💛", layout="centered")

apply_styles()


# =========================================================
# CONSTANTS
# =========================================================

BACKEND_URL = "http://127.0.0.1:8000"

# label shown in the dropdown  ->  name the backend understands
LANGUAGES = {
    "English": "English",
    "ಕನ್ನಡ": "Kannada",
    "हिन्दी": "Hindi",
    "தமிழ்": "Tamil",
    "తెలుగు": "Telugu",
    "मराठी": "Marathi",
}

CODE_TO_NAME = {
    "en": "English", "kn": "Kannada", "hi": "Hindi",
    "ta": "Tamil", "te": "Telugu", "mr": "Marathi",
}


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "page": "home",
    "language_label": "English",   # what the dropdown shows (e.g. ಕನ್ನಡ)
    "language": "English",         # what we send to the backend (e.g. Kannada)
    "messages": [],
    "morning_taken": False,
    "afternoon_taken": False,
    "night_taken": False,
    "voice_text": "",
    "voice_lang": "",              # language code Whisper used, e.g. "kn"
    "voice_reply": "",
    "voice_reply_lang": "",
    "voice_debug": None,
    "voice_warning": None,
    "voice_mismatch": False,
    # these were read with .get() but never declared or cleared, so a failed
    # recording left the PREVIOUS clip's numbers on screen
    "voice_probs": {},
    "voice_supported_mass": None,
    "voice_forced": False,
    "voice_low_confidence": False,
}

# every key a transcription owns -- reset as a group before each new attempt
VOICE_KEYS = [
    "voice_text", "voice_lang", "voice_reply", "voice_reply_lang",
    "voice_debug", "voice_warning", "voice_mismatch", "voice_probs",
    "voice_supported_mass", "voice_forced", "voice_low_confidence",
]


def reset_voice_state():
    for key in VOICE_KEYS:
        st.session_state[key] = defaults[key]

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# NAVIGATION
# =========================================================

def go_to(page):
    st.session_state.page = page
    st.rerun()


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="neravu-title">
        💛 Neravu
    </div>

    <div class="neravu-subtitle">
        Your multilingual AI companion
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("💛 Neravu")
    st.write("Choose a section")

    if st.button("🏠 Home", use_container_width=True):
        go_to("home")

    if st.button("💬 Chat", use_container_width=True):
        go_to("chat")

    if st.button("🎙️ Voice", use_container_width=True):
        go_to("voice")

    if st.button("💊 Medicines", use_container_width=True):
        go_to("medicines")

    if st.button("👨‍👩‍👧 Family & Caregiver", use_container_width=True):
        go_to("family")

    if st.button("🚨 Emergency", use_container_width=True):
        go_to("emergency")

    st.divider()

    st.subheader("🌐 Language")

    labels = list(LANGUAGES.keys())

    selected_label = st.selectbox(
        "Preferred language",
        labels,
        index=labels.index(st.session_state.language_label),
    )

    # FIX: store BOTH. The backend needs the English name ("Kannada"),
    # not the native label ("ಕನ್ನಡ"), otherwise it falls back to English.
    st.session_state.language_label = selected_label
    st.session_state.language = LANGUAGES[selected_label]


# =========================================================
# HOME
# =========================================================

if st.session_state.page == "home":

    st.title("👋 Welcome to Neravu")

    st.write(
        """
        Neravu is a simple AI companion designed to provide
        everyday assistance in Indian languages.
        """
    )

    st.divider()

    st.info(f"🌐 Current language: **{st.session_state.language}**")

    st.subheader("What can Neravu help with?")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("💬 Talk to Neravu", use_container_width=True):
            go_to("chat")

        if st.button("💊 My Medicines", use_container_width=True):
            go_to("medicines")

    with col2:
        if st.button("🎙️ Voice Conversation", use_container_width=True):
            go_to("voice")

        if st.button("👨‍👩‍👧 Family", use_container_width=True):
            go_to("family")

    st.divider()

    st.subheader("✨ Features")

    st.write("🗣️ Multilingual conversation")
    st.write("🎙️ Voice interaction")
    st.write("💊 Medicine reminders")
    st.write("🍛 Everyday assistance")
    st.write("❤️ Wellbeing support")
    st.write("👨‍👩‍👧 Family updates")

    st.divider()

    st.warning(
        "Neravu provides general information and assistance. "
        "It does not replace professional medical or emergency care."
    )


# =========================================================
# CHAT
# =========================================================

elif st.session_state.page == "chat":

    st.title("💬 Talk to Neravu")

    st.write(f"Responding in **{st.session_state.language}**")

    st.divider()

    if st.button("⬅️ Back to Home", key="chat_back"):
        go_to("home")

    if st.button("🧹 Clear Chat", key="clear_chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_message = st.chat_input("Type your message...")

    if user_message:

        st.session_state.messages.append(
            {"role": "user", "content": user_message}
        )

        with st.chat_message("user"):
            st.write(user_message)

        with st.chat_message("assistant"):

            with st.spinner("💛 Neravu is thinking..."):

                try:
                    response = send_message(user_message, st.session_state.language)

                    # FIX: backend key is "response" ("reply" kept as fallback)
                    bot_response = (
                        response.get("response")
                        or response.get("reply")
                        or "I could not generate a response."
                    )

                    st.write(bot_response)

                except Exception as e:
                    bot_response = "Sorry, I could not generate a response right now."
                    st.error(f"Backend error: {e}")

        st.session_state.messages.append(
            {"role": "assistant", "content": bot_response}
        )


# =========================================================
# VOICE
# =========================================================

elif st.session_state.page == "voice":

    st.title("🎙️ Voice Conversation")

    st.write(f"App language: **{st.session_state.language}**")

    st.divider()

    if st.button("⬅️ Back to Home", key="voice_back"):
        go_to("home")

    reply_in_spoken = st.checkbox(
        "Reply in the language I speak (auto-detected)",
        value=True,
        help="Turn off to always reply in the language chosen in the sidebar.",
    )

    force_app_language = st.checkbox(
        f"Force speech recognition in {st.session_state.language} "
        "(tick this if it keeps mishearing your language)",
        value=False,
    )

    st.subheader("🎤 Record your voice")

    st.write(
        """
        Click the microphone button, speak naturally,
        and stop recording when you are finished.
        """
    )

    audio_value = st.audio_input("🎤 Click here to record")

    if audio_value is not None:

        st.success("✅ Voice recorded successfully!")
        st.audio(audio_value)

        st.divider()

        # -------------------------------------------------
        # SPEECH -> TEXT
        # -------------------------------------------------

        if st.button(
            "📝 Convert Speech to Text",
            key="convert_voice",
            use_container_width=True,
        ):

            # drop the previous clip's text, warning and confidence numbers
            # before asking for new ones, so nothing stale survives a failure
            reset_voice_state()

            with st.spinner("🎧 Neravu is listening..."):

                try:
                    response = requests.post(
                        f"{BACKEND_URL}/transcribe",

                        # field name must be "file" (backend: file=File(...))
                        files={"file": ("voice.wav", audio_value.getvalue(), "audio/wav")},

                        # the sidebar language is a *hint* only; Whisper still
                        # detects what was really spoken
                        data={
                            "language": st.session_state.language,
                            "force": "1" if force_app_language else "0",
                        },

                        timeout=300,
                    )

                    # the backend now returns 4xx/5xx on failure, with the
                    # readable reason in the JSON body rather than raw text
                    try:
                        data = response.json()
                    except ValueError:
                        data = None

                    if data is None:
                        st.error(
                            f"Whisper server error {response.status_code}: "
                            f"{response.text[:200]}"
                        )

                    elif data.get("success"):
                        st.session_state.voice_text = data.get("text", "").strip()
                        st.session_state.voice_lang = data.get("language", "en")
                        st.session_state.voice_probs = data.get("probabilities", {})
                        st.session_state.voice_supported_mass = data.get("supported_mass")
                        st.session_state.voice_warning = data.get("warning")
                        st.session_state.voice_mismatch = data.get("hint_mismatch", False)
                        st.session_state.voice_forced = data.get("forced", False)
                        st.session_state.voice_low_confidence = data.get(
                            "low_confidence", False
                        )
                        st.success("✅ Speech converted successfully!")

                    else:
                        st.error(data.get("error", "Whisper could not process the audio."))

                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to Neravu backend.")
                    st.info("Start FastAPI first:  uvicorn main:app --reload")

                except Exception as e:
                    st.error(f"Voice error: {e}")

        # -------------------------------------------------
        # WHAT NERAVU HEARD (editable, so ASR mistakes can be fixed)
        # -------------------------------------------------

        if st.session_state.voice_text:

            st.divider()

            st.subheader("📝 What Neravu heard")

            st.session_state.voice_text = st.text_area(
                "You can correct the text before sending",
                value=st.session_state.voice_text,
                height=100,
            )

            heard_lang = CODE_TO_NAME.get(st.session_state.voice_lang, st.session_state.voice_lang)

            if st.session_state.get("voice_warning"):
                st.warning(st.session_state.voice_warning)

            if st.session_state.get("voice_mismatch"):
                if st.session_state.get("voice_forced"):
                    # this case used to be impossible to reach: forcing
                    # overwrote the detection, so a mis-heard forced clip
                    # reported no mismatch at all
                    st.warning(
                        f"Speech recognition was forced to "
                        f"**{st.session_state.language}**, but what I actually "
                        f"heard sounded more like **{heard_lang}**. Check the "
                        "text above carefully."
                    )
                else:
                    st.info(
                        f"I heard **{heard_lang}**, but the app language is "
                        f"**{st.session_state.language}**. If you actually spoke "
                        f"{st.session_state.language}, tick the 'Force speech "
                        "recognition' box above and record again."
                    )

            if st.session_state.get("voice_forced"):
                st.write(f"Transcribed as: **{st.session_state.language}** (forced)")
            else:
                st.write(f"Detected spoken language: **{heard_lang}**")

            probs = st.session_state.get("voice_probs")
            if probs:
                # These are raw Whisper probabilities over all 99 languages it
                # knows, filtered down to our six -- they do NOT add up to
                # 100%. Rendering them as bare percentages implied they did,
                # so the share that landed outside Neravu's languages is now
                # shown alongside them.
                st.caption(
                    "Whisper confidence: "
                    + ", ".join(
                        f"{CODE_TO_NAME.get(c, c)} {p:.0%}" for c, p in probs.items()
                    )
                )

                mass = st.session_state.get("voice_supported_mass")
                if mass is not None:
                    st.caption(
                        f"Only {mass:.0%} of Whisper's certainty landed on a "
                        "language Neravu supports; the rest went to languages "
                        "it does not handle."
                    )

            # ---------------------------------------------
            # ASK NERAVU
            # ---------------------------------------------

            st.divider()

            st.subheader("💬 Ask Neravu")

            if reply_in_spoken and st.session_state.voice_lang:
                reply_language = CODE_TO_NAME.get(
                    st.session_state.voice_lang, st.session_state.language
                )
            else:
                reply_language = st.session_state.language

            st.write(f"Neravu will reply in **{reply_language}**")

            if st.button(
                "💛 Get Neravu Response",
                key="ask_voice",
                use_container_width=True,
            ):

                with st.spinner("💛 Neravu is thinking..."):

                    try:
                        response = send_message(st.session_state.voice_text, reply_language)

                        bot_response = (
                            response.get("response")
                            or response.get("reply")
                            or "I could not generate a response."
                        )

                        st.session_state.messages.append(
                            {"role": "user", "content": st.session_state.voice_text}
                        )
                        st.session_state.messages.append(
                            {"role": "assistant", "content": bot_response}
                        )

                        st.session_state.voice_reply = bot_response
                        st.session_state.voice_reply_lang = reply_language
                        st.session_state.voice_debug = response.get("debug")

                    except Exception as e:
                        st.error(f"AI response error: {e}")

            # shown from session state so it survives reruns
            if st.session_state.voice_reply:
                st.subheader("💛 Neravu")
                st.write(st.session_state.voice_reply)

                dbg = st.session_state.voice_debug
                if dbg:
                    with st.expander("🔍 How Neravu understood this (debug)"):
                        st.write("**Path:**", dbg.get("path"))
                        st.write("**Text Neravu received:**", dbg.get("heard_text"))
                        st.write("**How it was understood:**", dbg.get("input_stage"))
                        st.write("**Your words in English:**", dbg.get("english_understanding"))
                        st.write("**Answer in English:**", dbg.get("english_answer"))

    st.divider()

    st.subheader("🌐 Supported Languages")

    st.write("🇮🇳 Kannada • Hindi • Tamil • Telugu • English • Marathi")


# =========================================================
# MEDICINES
# =========================================================

elif st.session_state.page == "medicines":

    st.title("💊 My Medicines")

    if st.button("⬅️ Back to Home", key="medicine_back"):
        go_to("home")

    st.divider()

    st.subheader("📅 Today's Medicines")

    slots = [
        ("morning", "### 🌅 Morning", "💊 Morning Medicine", "⏰ 9:00 AM", "Morning"),
        ("afternoon", "### ☀️ Afternoon", "💊 Afternoon Medicine", "⏰ 1:00 PM", "Afternoon"),
        ("night", "### 🌙 Night", "💊 Night Medicine", "⏰ 9:00 PM", "Night"),
    ]

    for slot, heading, name, time_label, short in slots:

        state_key = f"{slot}_taken"

        st.markdown(heading)
        st.write(name)
        st.write(time_label)

        if not st.session_state[state_key]:

            if st.button("✅ Mark as Taken", key=f"{slot}_taken_btn"):
                st.session_state[state_key] = True
                st.rerun()

        else:

            st.success(f"{short} medicine marked as taken.")

            if st.button("↩️ Undo", key=f"{slot}_undo"):
                st.session_state[state_key] = False
                st.rerun()

        st.divider()

    st.warning(
        "Medicine reminders are for tracking only. "
        "Neravu does not prescribe or change medication dosage."
    )


# =========================================================
# FAMILY
# =========================================================

elif st.session_state.page == "family":

    st.title("👨‍👩‍👧 Family & Caregiver")

    if st.button("⬅️ Back to Home", key="family_back"):
        go_to("home")

    st.divider()

    st.subheader("👨‍⚕️ Caregiver")

    st.info(
        """
        **Primary Caregiver**

        Available for assistance.
        """
    )

    st.divider()

    st.subheader("💊 Medicine Status")

    st.write("🌅 Morning — " + ("✅ Taken" if st.session_state.morning_taken else "⏳ Pending"))
    st.write("☀️ Afternoon — " + ("✅ Taken" if st.session_state.afternoon_taken else "⏳ Pending"))
    st.write("🌙 Night — " + ("✅ Taken" if st.session_state.night_taken else "⏳ Pending"))

    st.divider()

    st.subheader("🔔 Recent Updates")

    if st.session_state.messages:
        st.write("✓ Neravu conversation recorded.")
    else:
        st.write("No conversations yet.")

    st.success(
        "Family and caregiver updates can be connected "
        "to the database in the next integration stage."
    )


# =========================================================
# EMERGENCY
# =========================================================

elif st.session_state.page == "emergency":

    st.title("🚨 Emergency Help")

    if st.button("⬅️ Back to Home", key="emergency_back"):
        go_to("home")

    st.divider()

    st.error(
        """
        🚨 If you are experiencing a serious emergency,
        contact your local emergency service or a trusted
        family member/caregiver immediately.
        """
    )

    st.subheader("📞 Emergency Contacts")

    if st.button("📞 Contact Family", use_container_width=True):
        st.info("Connect this button to your family's emergency contact.")

    if st.button("📞 Contact Caregiver", use_container_width=True):
        st.info("Connect this button to your caregiver's contact.")

    st.divider()

    st.write(
        """
        **Important:**

        Neravu is an AI assistance system.
        It is not a replacement for emergency medical services.
        """
    )