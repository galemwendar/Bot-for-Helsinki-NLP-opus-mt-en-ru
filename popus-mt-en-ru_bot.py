import sys
import time
print("application starting...")

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-ru")
    model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-ru")

    while True:
        # Get input text from user
        text = input("Enter text to translate or type q to exit: ")
        
        if text == 'q':
            break
    
        # Tokenize the input text and generate the translation
        input_ids = tokenizer.encode(text, return_tensors="pt")
        output_ids = model.generate(input_ids, max_new_tokens=100)
        translation = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
        # Print the translation
        print("Translation: ", translation)

except Exception as e:
    # print the error message to the console
    print(f"Error: {str(e)}")
    # or write the error message to a log file
    with open("error.log", "a") as f:
        f.write(f"Error: {str(e)}\n")
    # prevent the app from closing by waiting for user input
    input("Press Enter to continue...")
    # or prevent the app from closing by pausing for a few seconds
    time.sleep(5)
    sys.exit(1)