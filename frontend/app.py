
import streamlit as st
import requests

from api import send_message
from styles import apply_styles


# =========================================================
# CONFIG
# =========================================================

st.set_page_config(
    page_title="Neravu",
    page_icon="💛",
    layout="centered"
)

apply_styles()


# =========================================================
# CONSTANTS
# =========================================================

BACKEND_URL = "http://127.0.0.1:8000"


# Language shown in the UI -> language name sent to FastAPI
LANGUAGES = {
    "English": "English",
    "ಕನ್ನಡ": "Kannada",
    "हिन्दी": "Hindi",
    "தமிழ்": "Tamil",
    "తెలుగు": "Telugu",
    "मराठी": "Marathi",
}


# Gemini transcription returns a language code/name depending
# on the backend implementation.
CODE_TO_NAME = {
    "en": "English",
    "kn": "Kannada",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "mr": "Marathi",
}


# =========================================================
# SESSION STATE
# =========================================================

defaults = {
    "page": "home",

    # Language
    "language_label": "English",
    "language": "English",

    # Chat
    "messages": [],

    # Medicines
    "morning_taken": False,
    "afternoon_taken": False,
    "night_taken": False,

    # Voice
    "voice_text": "",
    "voice_lang": "",
    "voice_reply": "",
    "voice_reply_lang": "",
    "voice_debug": None,
    "voice_error": None,
    "voice_transcription_time": None,
    "voice_urgent": False,
}


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
# HELPER FUNCTIONS
# =========================================================

def language_name_from_code(code):

    if not code:
        return st.session_state.language

    code = str(code).lower().strip()

    return CODE_TO_NAME.get(
        code,
        code
    )


def clear_voice_state():

    st.session_state.voice_text = ""
    st.session_state.voice_lang = ""
    st.session_state.voice_reply = ""
    st.session_state.voice_reply_lang = ""
    st.session_state.voice_debug = None
    st.session_state.voice_error = None
    st.session_state.voice_transcription_time = None
    st.session_state.voice_urgent = False


def transcribe_audio(audio_bytes):

    """
    Send recorded audio to FastAPI.

    FastAPI -> Gemini 3.5 Transcribe

    Gemini automatically detects the spoken language.
    """

    response = requests.post(

        f"{BACKEND_URL}/transcribe",

        files={
            "file": (
                "voice.wav",
                audio_bytes,
                "audio/wav"
            )
        },

        timeout=300,
    )

    return response


def get_ai_response(text, language):

    """
    Send text to FastAPI.

    FastAPI -> RAG -> Gemini 3.8 Flash
    """

    return send_message(
        text,
        language
    )


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

    if st.button(
        "🏠 Home",
        use_container_width=True
    ):
        go_to("home")

    if st.button(
        "💬 Chat",
        use_container_width=True
    ):
        go_to("chat")

    if st.button(
        "🎙️ Voice",
        use_container_width=True
    ):
        go_to("voice")

    if st.button(
        "💊 Medicines",
        use_container_width=True
    ):
        go_to("medicines")

    if st.button(
        "👨‍👩‍👧 Family & Caregiver",
        use_container_width=True
    ):
        go_to("family")

    if st.button(
        "🚨 Emergency",
        use_container_width=True
    ):
        go_to("emergency")

    st.divider()

    st.subheader("🌐 Language")

    labels = list(LANGUAGES.keys())

    selected_label = st.selectbox(
        "Preferred language",
        labels,
        index=labels.index(
            st.session_state.language_label
        ),
    )

    st.session_state.language_label = selected_label

    st.session_state.language = LANGUAGES[
        selected_label
    ]


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

    st.info(
        f"🌐 Current language: "
        f"**{st.session_state.language}**"
    )

    st.subheader(
        "What can Neravu help with?"
    )

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "💬 Talk to Neravu",
            use_container_width=True
        ):
            go_to("chat")

        if st.button(
            "💊 My Medicines",
            use_container_width=True
        ):
            go_to("medicines")

    with col2:

        if st.button(
            "🎙️ Voice Conversation",
            use_container_width=True
        ):
            go_to("voice")

        if st.button(
            "👨‍👩‍👧 Family",
            use_container_width=True
        ):
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

    st.write(
        f"Responding in **{st.session_state.language}**"
    )

    st.divider()

    if st.button(
        "⬅️ Back to Home",
        key="chat_back"
    ):
        go_to("home")

    if st.button(
        "🧹 Clear Chat",
        key="clear_chat"
    ):

        st.session_state.messages = []

        st.rerun()

    st.divider()

    # Display previous messages

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )

    user_message = st.chat_input(
        "Type your message..."
    )

    if user_message:

        # Add user message

        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_message
            }
        )

        with st.chat_message("user"):

            st.write(user_message)

        # Generate AI response

        with st.chat_message("assistant"):

            with st.spinner(
                "💛 Neravu is thinking..."
            ):

                try:

                    response = get_ai_response(
                        user_message,
                        st.session_state.language
                    )

                    bot_response = (
                        response.get("response")
                        or response.get("reply")
                        or "I could not generate a response."
                    )

                    st.write(
                        bot_response
                    )

                except Exception as e:

                    bot_response = (
                        "Sorry, I could not generate "
                        "a response right now."
                    )

                    st.error(
                        f"Backend error: {e}"
                    )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": bot_response
            }
        )


# =========================================================
# VOICE
# =========================================================

elif st.session_state.page == "voice":

    st.title("🎙️ Voice Conversation")

    st.write(
        "Speak naturally in English, Kannada, Hindi, "
        "Tamil, Telugu, or Marathi."
    )

    st.divider()

    if st.button(
        "⬅️ Back to Home",
        key="voice_back"
    ):
        go_to("home")


    # -----------------------------------------------------
    # LANGUAGE OPTION
    # -----------------------------------------------------

    reply_in_spoken = st.checkbox(
        "Reply in the language I speak",
        value=True,
        help=(
            "Gemini detects the language you speak. "
            "When enabled, Neravu replies in that language."
        ),
    )


    st.divider()


    # -----------------------------------------------------
    # RECORD AUDIO
    # -----------------------------------------------------

    st.subheader("🎤 Record your voice")

    st.write(
        """
        Click the microphone button, speak naturally,
        and stop recording when you are finished.
        """
    )


    audio_value = st.audio_input(
        "🎤 Click here to record"
    )


    if audio_value is not None:

        st.success(
            "✅ Voice recorded successfully!"
        )

        st.audio(
            audio_value
        )


        st.divider()


        # -------------------------------------------------
        # CLEAR PREVIOUS RESULT
        # -------------------------------------------------

        if st.button(
            "🔄 Record Again",
            key="record_again",
            use_container_width=True
        ):

            clear_voice_state()

            st.rerun()


        # -------------------------------------------------
        # SPEECH -> TEXT
        # -------------------------------------------------

        if st.button(
            "📝 Convert Speech to Text",
            key="convert_voice",
            use_container_width=True,
        ):

            clear_voice_state()

            with st.spinner(
                "🎧 Gemini is listening..."
            ):

                try:

                    response = transcribe_audio(
                        audio_value.getvalue()
                    )


                    # -------------------------------------
                    # SERVER ERROR
                    # -------------------------------------

                    if response.status_code != 200:

                        st.error(
                            "❌ Transcription server error:"
                        )

                        st.code(
                            response.text[:1000]
                        )

                    else:

                        data = response.json()


                        # ---------------------------------
                        # SUCCESS
                        # ---------------------------------

                        if data.get("success"):

                            transcript = (
                                data.get("text")
                                or data.get("transcript")
                                or ""
                            ).strip()


                            detected_language = (
                                data.get("language")
                                or data.get(
                                    "detected_language"
                                )
                                or ""
                            )


                            st.session_state.voice_text = (
                                transcript
                            )


                            st.session_state.voice_lang = (
                                detected_language
                            )


                            st.session_state.voice_transcription_time = (
                                data.get(
                                    "transcription_time"
                                )
                            )


                            st.success(
                                "✅ Speech converted successfully!"
                            )


                            # Show Gemini detection

                            detected_name = (
                                language_name_from_code(
                                    detected_language
                                )
                            )


                            st.info(
                                f"🌐 Gemini detected: "
                                f"**{detected_name}**"
                            )


                        else:

                            st.error(
                                data.get(
                                    "error",
                                    "Gemini could not process the audio."
                                )
                            )


                except requests.exceptions.ConnectionError:

                    st.error(
                        "❌ Cannot connect to Neravu backend."
                    )

                    st.info(
                        "Start FastAPI first:\n\n"
                        "`uvicorn main:app --reload --port 8000`"
                    )


                except requests.exceptions.Timeout:

                    st.error(
                        "⏱️ Gemini took too long to process "
                        "the audio. Please try a shorter recording."
                    )


                except Exception as e:

                    st.error(
                        f"Voice error: {e}"
                    )


        # -------------------------------------------------
        # WHAT NERAVU HEARD
        # -------------------------------------------------

        if st.session_state.voice_text:

            st.divider()

            st.subheader(
                "📝 What Neravu heard"
            )


            # Editable transcript

            edited_text = st.text_area(
                "You can correct the text before sending",
                value=st.session_state.voice_text,
                height=120,
                key="voice_transcript_editor",
            )


            # Keep edited text in session state

            st.session_state.voice_text = edited_text


            # ------------------------------------------------
            # DETECTED LANGUAGE
            # ------------------------------------------------

            detected_language = (
                st.session_state.voice_lang
            )

            detected_name = (
                language_name_from_code(
                    detected_language
                )
            )


            st.write(
                f"🌐 Detected spoken language: "
                f"**{detected_name}**"
            )


            # ------------------------------------------------
            # TRANSCRIPTION TIME
            # ------------------------------------------------

            transcription_time = (
                st.session_state.voice_transcription_time
            )


            if transcription_time:

                st.caption(
                    f"Speech processing time: "
                    f"{transcription_time:.2f} seconds"
                )


            # ------------------------------------------------
            # ASK NERAVU
            # ------------------------------------------------

            st.divider()

            st.subheader(
                "💬 Ask Neravu"
            )


            if reply_in_spoken:

                reply_language = detected_name

            else:

                reply_language = (
                    st.session_state.language
                )


            st.write(
                f"Neravu will reply in "
                f"**{reply_language}**"
            )


            if st.button(
                "💛 Get Neravu Response",
                key="ask_voice",
                use_container_width=True,
            ):

                with st.spinner(
                    "💛 Gemini is thinking..."
                ):

                    try:

                        response = get_ai_response(
                            st.session_state.voice_text,
                            reply_language
                        )


                        bot_response = (
                            response.get("response")
                            or response.get("reply")
                            or "I could not generate a response."
                        )


                        # ---------------------------------
                        # STORE CHAT
                        # ---------------------------------

                        st.session_state.messages.append(
                            {
                                "role": "user",
                                "content": (
                                    st.session_state.voice_text
                                )
                            }
                        )


                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": bot_response
                            }
                        )


                        # ---------------------------------
                        # STORE VOICE RESPONSE
                        # ---------------------------------

                        st.session_state.voice_reply = (
                            bot_response
                        )

                        st.session_state.voice_reply_lang = (
                            reply_language
                        )

                        st.session_state.voice_debug = (
                            response.get("debug")
                        )

                        st.session_state.voice_urgent = (
                            response.get(
                                "urgent",
                                False
                            )
                        )


                    except requests.exceptions.ConnectionError:

                        st.error(
                            "❌ Cannot connect to Neravu backend."
                        )


                    except Exception as e:

                        st.error(
                            f"AI response error: {e}"
                        )


            # ------------------------------------------------
            # SHOW NERAVU RESPONSE
            # ------------------------------------------------

            if st.session_state.voice_reply:

                st.divider()

                st.subheader(
                    "💛 Neravu"
                )


                if st.session_state.voice_urgent:

                    st.error(
                        "🚨 This may require urgent medical attention."
                    )


                st.write(
                    st.session_state.voice_reply
                )


                st.caption(
                    f"Response language: "
                    f"{st.session_state.voice_reply_lang}"
                )


                # -----------------------------------------
                # DEBUG INFORMATION
                # -----------------------------------------

                dbg = (
                    st.session_state.voice_debug
                )


                if dbg:

                    with st.expander(
                        "🔍 Technical information"
                    ):

                        if dbg.get("path"):

                            st.write(
                                "**Path:**",
                                dbg.get("path")
                            )

                        if dbg.get("heard_text"):

                            st.write(
                                "**Text received:**",
                                dbg.get("heard_text")
                            )

                        if dbg.get("input_stage"):

                            st.write(
                                "**Input stage:**",
                                dbg.get("input_stage")
                            )

                        if dbg.get("rag_used") is not None:

                            st.write(
                                "**RAG used:**",
                                dbg.get("rag_used")
                            )


    # -----------------------------------------------------
    # SUPPORTED LANGUAGES
    # -----------------------------------------------------

    st.divider()

    st.subheader(
        "🌐 Supported Languages"
    )

    st.write(
        "🇮🇳 English • Kannada • Hindi • Tamil • "
        "Telugu • Marathi"
    )


# =========================================================
# MEDICINES
# =========================================================

elif st.session_state.page == "medicines":

    st.title("💊 My Medicines")


    if st.button(
        "⬅️ Back to Home",
        key="medicine_back"
    ):
        go_to("home")


    st.divider()

    st.subheader(
        "📅 Today's Medicines"
    )


    slots = [

        (
            "morning",
            "### 🌅 Morning",
            "💊 Morning Medicine",
            "⏰ 9:00 AM",
            "Morning"
        ),

        (
            "afternoon",
            "### ☀️ Afternoon",
            "💊 Afternoon Medicine",
            "⏰ 1:00 PM",
            "Afternoon"
        ),

        (
            "night",
            "### 🌙 Night",
            "💊 Night Medicine",
            "⏰ 9:00 PM",
            "Night"
        ),
    ]


    for (
        slot,
        heading,
        name,
        time_label,
        short
    ) in slots:

        state_key = f"{slot}_taken"


        st.markdown(
            heading
        )

        st.write(
            name
        )

        st.write(
            time_label
        )


        if not st.session_state[state_key]:

            if st.button(
                "✅ Mark as Taken",
                key=f"{slot}_taken_btn"
            ):

                st.session_state[state_key] = True

                st.rerun()


        else:

            st.success(
                f"{short} medicine marked as taken."
            )


            if st.button(
                "↩️ Undo",
                key=f"{slot}_undo"
            ):

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

    st.title(
        "👨‍👩‍👧 Family & Caregiver"
    )


    if st.button(
        "⬅️ Back to Home",
        key="family_back"
    ):
        go_to("home")


    st.divider()


    st.subheader(
        "👨‍⚕️ Caregiver"
    )


    st.info(
        """
        **Primary Caregiver**

        Available for assistance.
        """
    )


    st.divider()


    st.subheader(
        "💊 Medicine Status"
    )


    st.write(
        "🌅 Morning — "
        + (
            "✅ Taken"
            if st.session_state.morning_taken
            else "⏳ Pending"
        )
    )


    st.write(
        "☀️ Afternoon — "
        + (
            "✅ Taken"
            if st.session_state.afternoon_taken
            else "⏳ Pending"
        )
    )


    st.write(
        "🌙 Night — "
        + (
            "✅ Taken"
            if st.session_state.night_taken
            else "⏳ Pending"
        )
    )


    st.divider()


    st.subheader(
        "🔔 Recent Updates"
    )


    if st.session_state.messages:

        st.write(
            "✓ Neravu conversation recorded."
        )

    else:

        st.write(
            "No conversations yet."
        )


    st.success(
        "Family and caregiver updates can be connected "
        "to the database in the next integration stage."
    )


# =========================================================
# EMERGENCY
# =========================================================

elif st.session_state.page == "emergency":

    st.title(
        "🚨 Emergency Help"
    )


    if st.button(
        "⬅️ Back to Home",
        key="emergency_back"
    ):
        go_to("home")


    st.divider()


    st.error(
        """
        🚨 If you are experiencing a serious emergency,
        contact your local emergency service or a trusted
        family member/caregiver immediately.
        """
    )


    st.subheader(
        "📞 Emergency Contacts"
    )


    if st.button(
        "📞 Contact Family",
        use_container_width=True
    ):

        st.info(
            "Connect this button to your family's emergency contact."
        )


    if st.button(
        "📞 Contact Caregiver",
        use_container_width=True
    ):

        st.info(
            "Connect this button to your caregiver's contact."
        )


    st.divider()


    st.write(
        """
        **Important:**

        Neravu is an AI assistance system.
        It is not a replacement for emergency medical services.
        """
    )

