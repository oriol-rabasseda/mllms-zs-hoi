from PIL import Image
import torch

class BaseModel:
    def __generate_messages(self, text_input, img_filepath) -> tuple[list, Image.Image]:
        pass

    def infer(self, text_input: str, img_filepath: str) -> str:
        pass

    def compute_score(self, score_lower_yes, score_upper_yes, score_lower_no, score_upper_no):
        scores = torch.FloatTensor([score_lower_yes, score_lower_no, score_upper_yes, score_upper_no])
        score = torch.nn.functional.softmax(scores, dim=0)
        return score[0].item() + score[2].item()
