from src.text_cleaner import clean_text


sample_text = """
PDTB113ZT


PNP 500 mA, 50 V


    resistor-equipped transistor
"""


cleaned_text = clean_text(sample_text)


print("====================================")
print("        TEXT CLEANER TEST")
print("====================================")

print("Original:")
print("------------------------------------")
print(sample_text)

print("Cleaned:")
print("------------------------------------")
print(cleaned_text)
print("------------------------------------")