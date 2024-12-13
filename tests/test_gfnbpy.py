import unittest
import bpy
from bpygfn import ActionEncoder


class TestActionEnc(unittest.TestCase):
    def setUp(self):
        # Clear existing objects
        bpy.ops.wm.read_factory_settings(use_empty=True)
        self.actions = ["one", "two", "three", "three"]

    def test_ctor(self):
        ae = ActionEncoder(action_vocab=self.actions)
        print(ae.action_to_idx)
        assert 1

    def test_enc_seq(self):
        encoder1 = ActionEncoder(
            max_sequence_length=30,
            action_vocab=["jump", "run", "walk", "duck", "idle"],
            positional_dim=4
        )
        
        # Test sequences
        actions = ["jump", "walk", "run", "idle", "jump"]
        
        # Demonstrate encodings
        one_hot = encoder1.one_hot_encode(actions)
        integer = encoder1.integer_encode(actions)
        positional = encoder1.positional_encode(actions)
        
        print("Vocabulary size:", encoder1.vocab_size)
        print("One-hot encoding shape:", one_hot.shape)
        print("Integer encoding shape:", integer.shape)
        print("Positional encoding shape:", positional.shape)
        print("One-hot dimension:", one_hot.shape[1])
        print("Positional embedding dimension:", positional.shape[1] - one_hot.shape[1])
        print(integer)
        # Show decoding
        decoded = encoder1.decode(integer)
        assert decoded == actions


if __name__ == "__main__":
    unittest.main()
