import numpy as np


class TravelToCityDataset:
    """A class to generate and store prompts and their correct and incorrect responses."""

    def __init__(self, n_examples):
        """Initialize the PromptGenerator with the number of examples."""
        self._places = [
            " Rome",
            " Paris",
            " Prague",
            " London",
            " Vienna",
            " Dublin",
            " Venice",
            " Lisbon",
            " Madrid",
            " Athens",
        ]
        self.clean_prompts = []
        self.correct_incorrects = []
        self.corrupted_prompts = []
        self._n_examples = n_examples
        self._generate_prompts()

    def _generate_prompts(self):
        """Generate the prompts and their correct and incorrect responses."""
        for _ in range(self._n_examples // 2):
            place1, place2 = np.random.choice(self._places, 2, replace=False)
            self.clean_prompts.extend(
                [
                    f"John was driving from{place1} to{place2}, when he arrived in",
                    f"Mary was driving from{place1} to{place2}, when she arrived in",
                ]
            )
            self.corrupted_prompts.extend(
                [
                    f"John was driving from{place2} to{place1}, when he arrived in",
                    f"Mary was driving from{place2} to{place1}, when she arrived in",
                ]
            )
            self.correct_incorrects.extend([(place2, place1), (place2, place1)])

    def print_prompts(self):
        """Print the prompts and their correct and incorrect responses."""
        print(f"Clean Prompts: {self.clean_prompts}\n")
        print(f"Corrupt Prompts: {self.corrupted_prompts}\n")
        print(f"Correct & Incorrect Responses: {self.correct_incorrects}\n")
