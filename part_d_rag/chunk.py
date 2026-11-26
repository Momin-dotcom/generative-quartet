import json
import os
from starter.bioasq import load_corpus, load_questions
CHUNK_SIZE=150 #maximum chunk size
OVERLAP=30 #30 word overlaping in chunks
STEP=CHUNK_SIZE-OVERLAP #step size for chunking
OUTPUTFILE="checkpoints/rag/chunks.jsonl"
def chunk_single_abstract(text):
    words=text.split() #split the abstract into words
    if len(words)==0:
        return []
    chunks=[]
    start=0
    while(True):
        end=start+CHUNK_SIZE #end from start to 150 words
        piece=words[start:end] #each piece of chunks
        chunks.append(" ".join(piece)) #join all the words in piece as a single chunk
        if end>=len(words): #if the end has reached end of the abstract break
            break
        start=start+STEP
    return chunks
def chunk_all(corpus):
    all_chunks=[]
    for abstract_id, text in corpus.items():# the dataset has abstract and its ids(traverse the whole dataset)
        for piece in chunk_single_abstract(text):# pass each abstract from chunk_single
            all_chunks.append({"abstract_id":abstract_id,"text":piece})# append each chunk with its abstract id to the all chunks list
    return all_chunks
def check_lengths(all_chunks):
    max=0
    for chunk in all_chunks:
        chunksize=len(chunk["text"].split())
        if chunksize>max:
            max=chunksize
    assert chunksize<=CHUNK_SIZE
    return max

if __name__ == "__main__":
    corpus = load_corpus()
    all_chunks = chunk_all(corpus)
    longest = check_lengths(all_chunks)

    # save the chunks
    os.makedirs("checkpoints/rag", exist_ok=True)
    with open(OUTPUTFILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # average number of relevant abstracts per test question 
    questions = load_questions("test", corpus)
    total_relevant = 0
    for q in questions:
        total_relevant = total_relevant + len(q["relevant_ids"])
    mean_relevant = total_relevant / len(questions)
    print()
    print("abstracts:", len(corpus))
    print("chunks:", len(all_chunks))
    print("longest chunk (words):", longest, "(must be 150 or less)")
    print("test questions:", len(questions))
    print("mean relevant abstracts per question:", round(mean_relevant, 2))




