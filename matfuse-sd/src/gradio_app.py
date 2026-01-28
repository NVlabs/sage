from pathlib import Path

import gradio as gr

from utils.inference_helpers import *

block = gr.Blocks().queue()
with block:
    gr.Markdown("# MatFuse")
    with gr.Tab("Generation"):
        with gr.Row():
            gr.Markdown("## Multi-Conditional Generation")
        with gr.Row():
            with gr.Column():
                input_image_emb = gr.Image(
                    source="upload",
                    type="pil",
                    label="Image",
                    shape=(1024, 1024),
                )
                gr.Examples(
                    [[x] for x in list(Path("sample_materials").glob("**/render.png"))],
                    inputs=input_image_emb,
                )
                input_image_palette = gr.Image(
                    source="upload",
                    type="pil",
                    label="Render Palette",
                    shape=(1024, 1024),
                )
                gr.Examples(
                    [[x] for x in list(Path("sample_materials").glob("**/render.png"))],
                    inputs=input_image_palette,
                )
                sketch = gr.Image(  # TODO Change with taking the sketch from an image using Canny
                    source="upload",
                    type="numpy",
                    label="Sketch",
                    image_mode="L",
                    shape=(256, 256),
                )
                gr.Examples(
                    [[x] for x in list(Path("sample_materials").glob("**/sketch.png"))],
                    inputs=sketch,
                )
                prompt = gr.Textbox(label="Prompt")
                gr.Examples(
                    [
                        ["A material of brick walls"],
                        ["A material of pebbles with grass textures"],
                        ["A material of asphalt and metal"],
                        ["A material of wood floor"],
                        ["A material of shiny wood floor"],
                    ],
                    inputs=prompt,
                )
                run_button = gr.Button(label="Run")
                with gr.Accordion("Advanced options", open=False):
                    num_samples = gr.Slider(
                        label="Images", minimum=1, maximum=12, value=1, step=1
                    )
                    image_resolution = gr.Slider(
                        label="Image Resolution",
                        minimum=256,
                        maximum=1024,
                        value=512,
                        step=64,
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=0,
                        maximum=20,
                        value=5.0,
                        step=0.1,
                    )
                    ddim_steps = gr.Slider(
                        label="Steps", minimum=1, maximum=100, value=50, step=1
                    )
                    seed = gr.Slider(
                        label="Seed",
                        minimum=-1,
                        maximum=2147483647,
                        step=1,
                        randomize=True,
                    )
                    eta = gr.Number(label="eta (DDIM)", value=0.0)
            with gr.Column():

                result_gallery = gr.Gallery(
                    label="Output", show_label=False, elem_id="gallery"
                ).style(grid=2, height="auto")
                gr.Markdown("###### Output Format: Sketch, Palette, Maps")
        conditions = [input_image_emb, input_image_palette, sketch, prompt]
        args = [num_samples, image_resolution, ddim_steps, seed, eta, guidance_scale]
        ips = [*conditions, *args]
        run_button.click(fn=run_generation, inputs=ips, outputs=[result_gallery])

    with gr.Tab("Editing"):
        gr.Markdown("## Material Editing")
        with gr.Row(elem_id="input_output"):
            with gr.Column(scale=2, elem_id="input"):
                gr.Markdown("## Input Maps")
                with gr.Row(elem_id="diff_rough"):
                    diff_map = gr.Image(
                        source="upload", type="pil", label="Diffuse", tool="sketch"
                    )
                    rough_map = gr.Image(
                        source="upload", type="pil", label="Roughness", tool="sketch"
                    )
                with gr.Row(elem_id="mask_diff_rough"):
                    mask_diff = gr.Checkbox(label="Mask Diffuse", default=False)
                    mask_rough = gr.Checkbox(label="Mask Roughness", default=False)
                with gr.Row(elem_id="examples_diff_rough"):
                    gr.Examples(
                        [
                            [x]
                            for x in list(
                                Path("sample_materials").glob("**/diffuse.png")
                            )
                        ],
                        inputs=diff_map,
                    )
                    gr.Examples(
                        [
                            [x]
                            for x in list(
                                Path("sample_materials").glob("**/roughness.png")
                            )
                        ],
                        inputs=rough_map,
                    )
                with gr.Row(elem_id="norm_spec"):
                    norm_map = gr.Image(
                        source="upload", type="pil", label="normal", tool="sketch"
                    )
                    spec_map = gr.Image(
                        source="upload", type="pil", label="specular", tool="sketch"
                    )
                with gr.Row(elem_id="mask_norm_spec"):
                    mask_norm = gr.Checkbox(label="Mask normal", default=False)
                    mask_spec = gr.Checkbox(label="Mask specular", default=False)
                with gr.Row(elem_id="examples_norm_spec"):
                    gr.Examples(
                        [
                            [x]
                            for x in list(
                                Path("sample_materials").glob("**/normal.png")
                            )
                        ],
                        inputs=norm_map,
                    )
                    gr.Examples(
                        [
                            [x]
                            for x in list(
                                Path("sample_materials").glob("**/specular.png")
                            )
                        ],
                        inputs=spec_map,
                    )

                gr.Markdown("## Input Conditions")
                with gr.Row(elem_id="conditions"):
                    input_image_palette = gr.Image(
                        source="upload",
                        type="pil",
                        label="Render Palette",
                        shape=(1024, 1024),
                    )
                    input_image_embed = gr.Image(
                        source="upload",
                        type="pil",
                        label="Render Embed",
                        shape=(1024, 1024),
                    )
                with gr.Row(elem_id="examples_conditions"):
                    gr.Examples(
                        [
                            [x]
                            for x in list(
                                Path("sample_materials").glob("**/render.png")
                            )
                        ],
                        inputs=input_image_palette,
                    )

                    gr.Examples(
                        [
                            [x]
                            for x in list(
                                Path("sample_materials").glob("**/render.png")
                            )
                        ],
                        inputs=input_image_embed,
                    )

                prompt = gr.Textbox(label="Prompt", lines=3)
                with gr.Accordion("Advanced options", open=False):
                    num_samples = gr.Slider(
                        label="Images", minimum=1, maximum=12, value=1, step=1
                    )
                    image_resolution = gr.Slider(
                        label="Image Resolution",
                        minimum=256,
                        maximum=1024,
                        value=512,
                        step=64,
                    )
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=0,
                        maximum=20,
                        value=5.0,
                        step=0.1,
                    )
                    ddim_steps = gr.Slider(
                        label="Steps", minimum=1, maximum=100, value=50, step=1
                    )
                    seed = gr.Slider(
                        label="Seed",
                        minimum=-1,
                        maximum=2147483647,
                        step=1,
                        randomize=True,
                    )
                    eta = gr.Number(label="eta (DDIM)", value=0.0)
                run_button = gr.Button(label="Run")

            with gr.Column(elem_id="output"):
                gr.Markdown("## Output")
                result_gallery = gr.Gallery(
                    label="Output", show_label=False, elem_id="gallery"
                )

        input_maps = [diff_map, norm_map, rough_map, spec_map]
        mask_maps = [mask_diff, mask_norm, mask_rough, mask_spec]
        conditions = [input_image_embed, prompt, input_image_palette]
        args = [image_resolution, seed, num_samples, guidance_scale, ddim_steps, eta]
        ips = [*input_maps, *mask_maps, *conditions, *args]
        run_button.click(fn=run_editing, inputs=ips, outputs=[result_gallery])

block.launch(server_name="0.0.0.0", share=True)
