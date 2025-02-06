from typing import *
import torch
import torch.nn as nn
from PIL import Image

from submodules.TRELLIS.trellis.modules import sparse as sp
from submodules.TRELLIS.trellis.pipelines import samplers
from submodules.TRELLIS.trellis.pipelines import TrellisImageTo3DPipeline
from ..utils.sparse_utils import get_voxels, convert_voxels_to_pcd

class PointCloudTo3DPipeline(TrellisImageTo3DPipeline):
    def __init__(
        self,
        models = None,
        sparse_structure_sampler = None,
        slat_sampler = None,
        slat_normalization: dict = None,
        image_cond_model: str = None,
    ):
        if models is None:
            return

        super.__init__(self, models, sparse_structure_sampler, slat_sampler, slat_normalization, image_cond_model)

    @staticmethod
    def from_pretrained(path: str) -> "PointCloudTo3DPipeline":

        pipeline = super(PointCloudTo3DPipeline, PointCloudTo3DPipeline).from_pretrained(path)
        new_pipeline = PointCloudTo3DPipeline()
        new_pipeline.__dict__ = pipeline.__dict__
        args = pipeline._pretrained_args

        new_pipeline.sparse_structure_sampler = getattr(samplers, args['sparse_structure_sampler']['name'])(**args['sparse_structure_sampler']['args'])
        new_pipeline.sparse_structure_sampler_params = args['sparse_structure_sampler']['params']

        new_pipeline.slat_sampler = getattr(samplers, args['slat_sampler']['name'])(**args['slat_sampler']['args'])
        new_pipeline.slat_sampler_params = args['slat_sampler']['params']

        new_pipeline.slat_normalization = args['slat_normalization']

        new_pipeline._init_image_cond_model(args['image_cond_model'])

        return new_pipeline

    def get_sparse_structure(self, input_path):

        mapping = {}

        voxels, transformation_info = get_voxels(input_path)

        voxel_tensor = torch.tensor(voxels, dtype=torch.float32)
        print(voxel_tensor.shape)
        
        voxel_tensor = voxel_tensor.unsqueeze(0).unsqueeze(0).to(self.device)

        coords = torch.argwhere(voxel_tensor)[:, [0, 2, 3, 4]].int()

        mapping["voxels"] = convert_voxels_to_pcd(voxels)
        mapping["transform"] = transformation_info

        return coords, mapping

    @torch.no_grad()
    def run_sparse_structure(
        self,
        images: List[Image.Image],
        num_samples: int = 1,
        input_path: str = None,
        seed: int = 42,
        slat_sampler_params: dict = {},
        formats: List[str] = ['mesh', 'gaussian', 'radiance_field'],
        preprocess_image: bool = True,
        mode: Literal['stochastic', 'multidiffusion'] = 'stochastic',
    ) -> dict:

        torch.manual_seed(seed)
        
        if preprocess_image:
            for i in range(len(images)):
                images[i] = self.preprocess_image(images[i])
            
        coords, mapping = self.get_sparse_structure(input_path)

        cond = self.get_cond(images)
        cond['neg_cond'] = cond['neg_cond'][:1]

        slat_steps = {**self.slat_sampler_params, **slat_sampler_params}.get('steps')
        with self.inject_sampler_multi_image('slat_sampler', len(images), slat_steps, mode=mode):
            slat = self.sample_slat(cond, coords, slat_sampler_params)

        return self.decode_slat(slat, formats), mapping


