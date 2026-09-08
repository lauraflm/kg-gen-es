import argparse
import nltk


# Ensure the punkt tokenizer is downloaded
def ensure_nltk_resource(resource_path, resource_name):
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(resource_name, quiet=True)


ensure_nltk_resource("tokenizers/punkt", "punkt")
ensure_nltk_resource("tokenizers/punkt_tab", "punkt_tab")

# Map language codes to the language names punkt expects. punkt ships a
# separate sentence tokenizer per language; the wrong one splits on the
# wrong abbreviations and misses Spanish's inverted "¿"/"¡" openers, which
# cuts sentences mid-clause before the LLM ever sees them.
_PUNKT_LANGUAGES = {
    "en": "english",
    "es": "spanish",
}

_DEFAULT_PUNKT_LANGUAGE = "english"


def chunk_text(text: str, max_chunk_size=500, language: str = "en") -> list[str]:
    """
    Chunk text by sentence, respecting a maximum chunk size.
    Falls back to word-based chunking if a single sentence is too large.

    :param text: The text to chunk.
    :param max_chunk_size: The maximum length (in characters) of any chunk.
    :param language: Language code used to pick punkt's sentence tokenizer.
        Unknown codes fall back to English.
    :return: A list of text chunks.
    """
    # Step 1: Split text into sentences
    punkt_language = _PUNKT_LANGUAGES.get(language, _DEFAULT_PUNKT_LANGUAGE)
    sentences = nltk.sent_tokenize(text, language=punkt_language)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding this sentence stays within the limit, append it.
        if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            # If the current chunk has some content, push it and start a new one.
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""

            # Check if the sentence itself is larger than the limit.
            # If yes, chunk it by words (fallback).
            if len(sentence) > max_chunk_size:
                words = sentence.split()
                temp_chunk = ""

                for word in words:
                    if len(temp_chunk) + len(word) + 1 <= max_chunk_size:
                        temp_chunk += word + " "
                    else:
                        chunks.append(temp_chunk.strip())
                        temp_chunk = word + " "

                # Add the leftover if any
                if temp_chunk:
                    chunks.append(temp_chunk.strip())
            else:
                # If the sentence is smaller than max_chunk_size, just start a new chunk with it.
                current_chunk = sentence + " "

    # If there's a leftover chunk that didn't get pushed, add it
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def main():
    parser = argparse.ArgumentParser(
        description="Chunk large text into smaller pieces while respecting sentence boundaries."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Path to the input text file. If omitted, reads from stdin.",
        default=None,
    )
    parser.add_argument(
        "--max_chunk_size",
        type=int,
        help="Maximum chunk size in characters (default=500).",
        default=500,
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Language code for sentence splitting (default=en).",
        default="en",
    )
    args = parser.parse_args()

    # Read the input text
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        import sys

        text = sys.stdin.read()

    # Chunk the text
    result_chunks = chunk_text(
        text, max_chunk_size=args.max_chunk_size, language=args.language
    )

    # Print or otherwise process the chunks
    for i, chunk in enumerate(result_chunks, start=1):
        print(f"--- Chunk {i} (length {len(chunk)}): ---")
        print(chunk)
        print()


if __name__ == "__main__":
    main()
