from chat_cembrowski.data.serialization import load_papers_from_json, save_paper

if __name__ == "__main__":
    for paper in load_papers_from_json():
        paper.processed = False
        save_paper(paper)
