from translator import translate_to_language

text = "I have a headache and fever."

languages = [
    "Kannada",
    "Telugu",
    "Tamil",
    "Hindi",
    "Marathi"
]

for language in languages:
    print("\n================================")
    print("TARGET:", language)
    print("================================")

    result = translate_to_language(text, language)

    print("RESULT:")
    print(result)