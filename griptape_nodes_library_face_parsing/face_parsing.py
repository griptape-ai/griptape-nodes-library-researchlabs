import logging

from face_parsing_model import (
    CLASS_NAMES,
    colorize_overlay,
    load_model,
    predict_labels,
    read_image,
    selection_mask,
    to_png_bytes,
)
from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape_nodes.exe_types.core_types import NodeError, Parameter, ParameterMode
from griptape_nodes.exe_types.node_types import AsyncResult, SuccessFailureNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter
from griptape_nodes.exe_types.param_components.project_file_parameter import ProjectFileParameter
from griptape_nodes.exe_types.param_types.parameter_image import ParameterImage
from griptape_nodes.traits.multi_options import MultiOptions

logger = logging.getLogger("face_parsing_library")

MODEL_REPO_IDS = ["vivym/face-parsing-bisenet"]

# Background is left out; select everything and invert to get it.
PART_CHOICES = CLASS_NAMES[1:]


class FaceParsing(SuccessFailureNode):
    """Create a mask of the selected face parts (skin, hair, eyes, and more) and a color preview of every part."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self._model_repo_parameter = HuggingFaceRepoParameter(
            self,
            repo_ids=MODEL_REPO_IDS,
            parameter_name="model",
        )
        self._model_repo_parameter.add_input_parameters()

        self.add_parameter(
            ParameterImage(
                name="image",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                tooltip="Image containing a face to segment.",
            )
        )
        self.add_parameter(
            Parameter(
                name="parts",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="list",
                default_value=["Skin"],
                tooltip="Face parts to include in the mask. Leave empty to produce only the overlay.",
                traits={MultiOptions(choices=PART_CHOICES, placeholder="Select face parts...")},
            )
        )
        self.add_parameter(
            Parameter(
                name="invert",
                allowed_modes={ParameterMode.INPUT, ParameterMode.PROPERTY},
                type="bool",
                default_value=False,
                tooltip="Swap the mask so the selected parts are black and everything else is white.",
            )
        )

        self.add_parameter(
            ParameterImage(
                name="mask",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="White where a selected part appears, black elsewhere.",
            )
        )

        self.add_parameter(
            ParameterImage(
                name="overlay",
                allowed_modes={ParameterMode.OUTPUT},
                tooltip="Preview of the input image with every detected face part tinted in its own color.",
            )
        )

        self._output_file = ProjectFileParameter(node=self, name="output_file", default_filename="face_mask.png")
        self._output_file.add_parameter()
        self._overlay_file = ProjectFileParameter(node=self, name="overlay_file", default_filename="face_overlay.png")
        self._overlay_file.add_parameter()

        # Status parameters MUST be the last thing added in __init__
        self._create_status_parameters()

    def validate_before_node_run(self) -> list[Exception] | None:
        errors: list[Exception] = []
        hf_errors = self._model_repo_parameter.validate_before_node_run()
        if hf_errors:
            errors.extend(hf_errors)
        if not isinstance(self.get_parameter_value("image"), (ImageArtifact, ImageUrlArtifact)):
            errors.append(NodeError("Connect an image to 'Image'."))
        return errors or None

    def process(self) -> AsyncResult[None]:
        yield lambda: self._run_inference()

    def _run_inference(self) -> None:
        self._set_status_results(was_successful=False, result_details="Running face parsing...")
        try:
            repo_id, revision = self._model_repo_parameter.get_repo_revision()
            image = read_image(self.get_parameter_value("image"))
            # get_parameter_value falls back to the parameter default; parameter_values only holds edited values.
            parts = self.get_parameter_list_value("parts")
            class_indices = [CLASS_NAMES.index(name) for name in parts if name in CLASS_NAMES]
            invert = bool(self.get_parameter_value("invert"))

            net = load_model(repo_id, revision)
            labels = predict_labels(net, image)
            overlay = colorize_overlay(image, labels)
            overlay_saved = self._overlay_file.build_file().write_bytes(to_png_bytes(overlay))

            # With no parts selected there is nothing to mask, so only the overlay is produced.
            mask_saved = None
            if class_indices:
                mask = selection_mask(labels, class_indices, invert)
                mask_saved = self._output_file.build_file().write_bytes(to_png_bytes(mask))
        except Exception as e:
            error = e if isinstance(e, NodeError) else NodeError(f"Face parsing failed: {e}")
            self._set_status_results(was_successful=False, result_details=str(error))
            self._handle_failure_exception(error)
            return

        self.parameter_output_values["overlay"] = ImageUrlArtifact(overlay_saved.location)
        self.parameter_output_values["mask"] = ImageUrlArtifact(mask_saved.location) if mask_saved else None
        if mask_saved:
            self._set_status_results(was_successful=True, result_details="Face parsing completed.")
        else:
            self._set_status_results(
                was_successful=True, result_details="No parts selected, so only the overlay was created."
            )
