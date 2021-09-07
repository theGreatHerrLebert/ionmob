import unittest
from preprocess.helpers import align_annotation, sequence_to_tokens, fit_tokenizer


class TestHelpers(unittest.TestCase):

    def test_align_annotation(self):
        seqs = ['_ADM(Oxidation (M))VIEAVFEDLSLK_', '_AM(Oxidation (M))GIM(Oxidation (M))NSFVNDIFER_','_AMGIMNSFVNDIFER_']
        result = ['_ADM(ox)VIEAVFEDLSLK_', '_AM(ox)GIM(ox)NSFVNDIFER_', '_AMGIMNSFVNDIFER_']
        replaced = [align_annotation(seq) for seq in seqs]
        self.assertEqual(replaced, result)

    def test_sequence_to_tokens(self):
        sequence = '_AM(ox)GIM(ox)NSFVNDIFER_'
        tokens = sequence_to_tokens(sequence)
        result = ['#-A', 'M-OX', 'G', 'I', 'M-OX', 'N', 'S', 'F', 'V', 'N', 'D', 'I', 'F', 'E', 'R-#']
        self.assertEqual(tokens, result)

    def test_tokenizer(self):
        sequence = '_AM(ox)GIM(ox)NSFVNDIFER_'
        tokens = ['#-A', 'M-OX', 'G', 'I', 'M-OX', 'N', 'S', 'F', 'V', 'N', 'D', 'I', 'F', 'E', 'R-#']
        tokenizer = fit_tokenizer([tokens])
        self.assertIs(len(tokenizer.word_index), 11)


if __name__ == '__main__':
    unittest.main()
