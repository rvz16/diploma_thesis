"""Guard model loading and thinking control without downloading model weights."""

import unittest
from unittest.mock import MagicMock, patch

from fc_constrained import ConstrainedFC


class ModelLoadingTests(unittest.TestCase):
    def test_text_only_multimodal_loader_is_explicit(self):
        loader = MagicMock()
        with patch("fc_constrained.AutoTokenizer.from_pretrained"), \
             patch("transformers.AutoModelForMultimodalLM", loader, create=True), \
             patch.object(ConstrainedFC, "_build_id2str", return_value=[]):
            ConstrainedFC("fixture", model_loader="multimodal", enable_thinking=False)
        loader.from_pretrained.assert_called_once()
        self.assertEqual(loader.from_pretrained.call_args.args, ("fixture",))

    def test_disable_thinking_reaches_the_chat_template(self):
        decoder = ConstrainedFC.__new__(ConstrainedFC)
        decoder.enable_thinking = False
        decoder.tok = MagicMock()
        decoder.build_prefix("system", "user")
        self.assertIs(decoder.tok.apply_chat_template.call_args.kwargs["enable_thinking"], False)

    def test_default_preserves_legacy_chat_template_options(self):
        decoder = ConstrainedFC.__new__(ConstrainedFC)
        decoder.tok = MagicMock()
        decoder.build_prefix("system", "user")
        self.assertNotIn("enable_thinking", decoder.tok.apply_chat_template.call_args.kwargs)


if __name__ == "__main__":
    unittest.main()
