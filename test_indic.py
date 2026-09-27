import sys
try:
    from translate_indic import translate
    out = translate("माझं नाव श्रीपाद आहे, मी एका EXTC division मध्ये आहे.", "mr", "en")
    print(out)
except Exception as e:
    import traceback
    traceback.print_exc()
