"""
Created on Mon Oct 26 20:47:34 2020

@author: Christian Murphy
"""

import math

import numpy as np
import torch


def preprocess(image):
    # [0, 1] => [-1, 1]
    return image * 2 - 1


def deprocess(image):
    # [-1, 1] => [0, 1]
    return (image + 1) / 2


# Log a tensor and normalize it.
def logTensor(tensor):
    return (torch.log(torch.add(tensor, 0.01)) - torch.log(0.01)) / (
        torch.log(1.01) - torch.log(0.01)
    )


def rand_range(shape, low, high, dtype=torch.float32):
    return (high - low) * torch.rand(shape, dtype=dtype)


def randn_range(shape, mean, std, dtype=torch.float32):
    return torch.randn(shape, dtype=dtype) * std + mean


# Generate a random direction on the upper hemisphere with gaps on the top and bottom of Hemisphere. Equation is described in the Global Illumination Compendium (19a)
def generate_normalized_random_direction(
    batchSize, nbRenderings, lowEps=0.001, highEps=0.05
):
    r1 = rand_range(
        (batchSize, nbRenderings, 1), 0.0 + lowEps, 1.0 - highEps, dtype=torch.float32
    )
    r2 = torch.rand([batchSize, nbRenderings, 1], dtype=torch.float32)
    r = torch.sqrt(r1)
    phi = 2 * math.pi * r2
    # min alpha = atan(sqrt(1-r^2)/r)
    x = r * torch.cos(phi)
    y = r * torch.sin(phi)
    z = torch.sqrt(1.0 - torch.square(r))
    finalVec = torch.cat(
        [x, y, z], axis=-1
    )  # Dimension here should be [batchSize,nbRenderings, 3]
    return finalVec


# Remove the gamma of a vector
def removeGamma(tensor):
    return torch.pow(tensor, 2.2)


# Add gamma to a vector
def addGamma(tensor):
    return torch.pow(tensor, 0.4545)


# Normalizes a tensor troughout the Channels dimension (BatchSize, Width, Height, Channels)
# Keeps 4th dimension to 1. Output will be (BatchSize, Width, Height, 1).
def normalize(tensor):
    Length = torch.sqrt(torch.sum(torch.square(tensor), dim=-1, keepdim=True))
    return torch.div(tensor, Length)


# Generate a distance to compute for the specular renderings (as position is important for this kind of renderings)
def generate_distance(batchSize, nbRenderings):
    gaussian = randn_range(
        [batchSize, nbRenderings, 1], 0.5, 0.75, dtype=torch.float32
    )  # parameters chosen empirically to have a nice distance from a -1;1 surface.
    return torch.exp(gaussian)


# Very small lamp attenuation
def lampAttenuation(distance):
    DISTANCE_ATTENUATION_MULT = 0.001
    return 1.0 / (1.0 + DISTANCE_ATTENUATION_MULT * torch.square(distance))


# Physically based lamp attenuation
def lampAttenuation_pbr(distance):
    return 1.0 / torch.square(distance)


# Clip values between min an max
def squeezeValues(tensor, min, max):
    return torch.clamp(tensor, min, max)


def DotProduct(tensorA, tensorB):
    return torch.sum(torch.mul(tensorA, tensorB), dim=-1, keepdim=True)


# Generate an array grid between -1;1 to act as the "coordconv" input layer (see coordconv paper)
def generateCoords(inputShape):
    crop_size = inputShape[-2]
    firstDim = inputShape[0]

    Xcoords = torch.unsqueeze(torch.linspace(-1.0, 1.0, crop_size), axis=0)
    Xcoords = Xcoords.repeat(crop_size, 1)
    Ycoords = -1 * Xcoords.T  # put -1 in the bottom of the table
    Xcoords = torch.unsqueeze(Xcoords, axis=-1)
    Ycoords = torch.unsqueeze(Ycoords, axis=-1)
    coords = torch.cat([Xcoords, Ycoords], axis=-1)
    coords = torch.unsqueeze(
        coords, axis=0
    )  # Add dimension to support batch size and nbRenderings should now be [1, 256, 256, 2].
    coords = coords.repeat(
        firstDim, 1, 1, 1
    )  # Add the proper dimension here for concat
    return coords


# Generate an array grid between -1;1 to act as each pixel position for the rendering.
def generateSurfaceArray(crop_size, pixelsToAdd=0):
    totalSize = crop_size + (pixelsToAdd * 2)
    surfaceArray = []
    XsurfaceArray = torch.unsqueeze(torch.linspace(-1.0, 1.0, totalSize), axis=0)
    XsurfaceArray = XsurfaceArray.repeat(totalSize, 1)
    YsurfaceArray = -1 * XsurfaceArray.T  # put -1 in the bottom of the table
    XsurfaceArray = torch.unsqueeze(XsurfaceArray, axis=-1)
    YsurfaceArray = torch.unsqueeze(YsurfaceArray, axis=-1)

    surfaceArray = torch.cat(
        [
            XsurfaceArray,
            YsurfaceArray,
            torch.zeros([totalSize, totalSize, 1], dtype=torch.float32),
        ],
        axis=-1,
    )
    surfaceArray = torch.unsqueeze(
        torch.unsqueeze(surfaceArray, axis=0), axis=0
    )  # Add dimension to support batch size and nbRenderings
    return surfaceArray


# create small variation to be added to the positions of lights or camera.
def jitterPosAround(batchSize, nbRenderings, posTensor, mean=0.0, stddev=0.03):
    randomPerturbation = torch.clamp(
        randn_range(
            [batchSize, nbRenderings, 1, 1, 1, 3], mean, stddev, dtype=torch.float32
        ),
        -0.24,
        0.24,
    )  # Clip here how far it can go to 8 * stddev to avoid negative values on view or light ( Z minimum value is 0.3)
    return posTensor + randomPerturbation


# Adds a little bit of noise
def addNoise(renderings):
    shape = renderings.shape
    stddevNoise = torch.exp(randn_range(1, mean=torch.log(0.005), stddev=0.3))
    noise = randn_range(shape, mean=0.0, std=stddevNoise)
    return renderings + noise


class GGXRenderer:
    includeDiffuse = True

    def __init__(self, includeDiffuse=True):
        self.includeDiffuse = includeDiffuse

    # Compute the diffuse part of the equation
    def compute_diffuse(self, diffuse, specular):
        return diffuse * (1.0 - specular) / math.pi

    # Compute the distribution function D driving the statistical orientation of the micro facets.
    def compute_distribution(self, roughness, NdotH):
        alpha = torch.square(roughness)
        underD = 1 / torch.clamp(
            (torch.square(NdotH) * (torch.square(alpha) - 1.0) + 1.0), min=0.001
        )
        return torch.square(alpha * underD) / math.pi

    # Compute the fresnel approximation F
    def compute_fresnel(self, specular, VdotH):
        sphg = torch.pow(2.0, ((-5.55473 * VdotH) - 6.98316) * VdotH)
        return specular + (1.0 - specular) * sphg

    # Compute the Geometry term (also called shadowing and masking term) G taking into account how microfacets can shadow each other.
    def compute_geometry(self, roughness, NdotL, NdotV):
        return self.G1(NdotL, torch.square(roughness) / 2) * self.G1(
            NdotV, torch.square(roughness) / 2
        )

    def G1(self, NdotW, k):
        return 1.0 / torch.clamp((NdotW * (1.0 - k) + k), min=0.001)

    # This computes the equations of Cook-Torrance for a BRDF without taking light power etc... into account.
    def calculateBRDF(
        self, svbrdf, wiNorm, woNorm, currentConeTargetPos, currentLightPos, multiLight
    ):

        h = normalize(torch.add(wiNorm, woNorm) / 2.0)
        # Put all the parameter values between 0 and 1 except the normal as they should be used between -1 and 1 (as they express a direction in a 360° sphere)
        normals = torch.unsqueeze(svbrdf[:, :, :, 3:6], axis=1)
        normals = normalize(normals)
        diffuse = torch.unsqueeze(
            squeezeValues(deprocess(svbrdf[:, :, :, 0:3]), 0.0, 1.0), dim=1
        )
        roughness = torch.unsqueeze(
            squeezeValues(deprocess(svbrdf[:, :, :, 6:9]), 0.0, 1.0), dim=1
        )
        specular = torch.unsqueeze(
            squeezeValues(deprocess(svbrdf[:, :, :, 9:12]), 0.0, 1.0), dim=1
        )
        # Avoid roughness = 0 to avoid division by 0

        roughness = torch.clamp(roughness, min=0.001)

        # If we have multiple lights to render, add a dimension to handle it.
        if multiLight:
            diffuse = torch.unsqueeze(diffuse, dim=1)
            normals = torch.unsqueeze(normals, dim=1)
            specular = torch.unsqueeze(specular, dim=1)
            roughness = torch.unsqueeze(roughness, dim=1)

        NdotH = DotProduct(normals, h)
        NdotH[NdotH != NdotH] = 0

        NdotL = DotProduct(normals, wiNorm)
        NdotL[NdotL != NdotL] = 0

        NdotV = DotProduct(normals, woNorm)
        NdotV[NdotV != NdotV] = 0

        VdotH = DotProduct(woNorm, h)

        diffuse_rendered = self.compute_diffuse(diffuse, specular)
        D_rendered = self.compute_distribution(roughness, torch.clamp(NdotH, min=0.0))
        G_rendered = self.compute_geometry(
            roughness, torch.clamp(NdotL, min=0.0), torch.clamp(NdotV, min=0.0)
        )
        F_rendered = self.compute_fresnel(specular, torch.clamp(VdotH, min=0.0))

        specular_rendered = F_rendered * (G_rendered * D_rendered * 0.25)
        result = specular_rendered

        # Add the diffuse part of the rendering if required.
        if self.includeDiffuse:
            result = result + diffuse_rendered
        return result, NdotL

    def render(
        self,
        svbrdf,
        wi,
        wo,
        currentConeTargetPos,
        tensorboard="",
        multiLight=False,
        currentLightPos=None,
        lossRendering=True,
        isAmbient=False,
        useAugmentation=True,
    ):
        wiNorm = normalize(wi)
        woNorm = normalize(wo)

        # Calculate how the image should look like with completely neutral lighting
        result, NdotL = self.calculateBRDF(
            svbrdf, wiNorm, woNorm, currentConeTargetPos, currentLightPos, multiLight
        )
        resultShape = result.shape
        lampIntensity = 1.5

        # Add lighting effects
        if not currentConeTargetPos is None:
            # If we want a cone light (to have a flash fall off effect)
            currentConeTargetDir = (
                currentLightPos - currentConeTargetPos
            )  # currentLightPos should never be None when currentConeTargetPos isn't
            coneTargetNorm = normalize(currentConeTargetDir)
            distanceToConeCenter = torch.maximum(
                0.0, DotProduct(wiNorm, coneTargetNorm)
            )
        if not lossRendering:
            # If we are not rendering for the loss
            if not isAmbient:
                if useAugmentation:
                    # The augmentations will allow different light power and exposures
                    stdDevWholeBatch = torch.exp(torch.randn((), mean=-2.0, stddev=0.5))
                    # add a normal distribution to the stddev so that sometimes in a minibatch all the images are consistant and sometimes crazy.
                    lampIntensity = torch.abs(
                        torch.randn(
                            (resultShape[0], resultShape[1], 1, 1, 1),
                            mean=10.0,
                            stddev=stdDevWholeBatch,
                        )
                    )  # Creates a different lighting condition for each shot of the nbRenderings Check for over exposure in renderings
                    # autoExposure
                    autoExposure = torch.exp(
                        torch.randn((), mean=np.log(1.5), stddev=0.4)
                    )
                    lampIntensity = lampIntensity * autoExposure
                else:
                    lampIntensity = torch.reshape(
                        torch.FloatTensor(13.0), [1, 1, 1, 1, 1]
                    )  # Look at the renderings when not using augmentations
            else:
                # If this uses ambient lighting we use much small light values to not burn everything.
                if useAugmentation:
                    lampIntensity = torch.exp(
                        torch.randn(
                            (resultShape[0], 1, 1, 1, 1),
                            mean=torch.log(0.15),
                            stddev=0.5,
                        )
                    )  # No need to make it change for each rendering.
                else:
                    lampIntensity = torch.reshape(
                        torch.FloatTensor(0.15), [1, 1, 1, 1, 1]
                    )
            # Handle light white balance if we want to vary it..
            if useAugmentation and not isAmbient:
                whiteBalance = torch.abs(
                    torch.randn(
                        [resultShape[0], resultShape[1], 1, 1, 3], mean=1.0, stddev=0.03
                    )
                )
                lampIntensity = lampIntensity * whiteBalance

            if multiLight:
                lampIntensity = torch.unsqueeze(
                    lampIntensity, axis=2
                )  # add a constant dim if using multiLight
        lampFactor = lampIntensity * math.pi

        if not isAmbient:
            if not lossRendering:
                # Take into accound the light distance (and the quadratic reduction of power)
                lampDistance = torch.sqrt(
                    torch.sum(torch.square(wi), axis=-1, keep_dims=True)
                )

                lampFactor = lampFactor * lampAttenuation_pbr(lampDistance)
            if not currentConeTargetPos is None:
                # Change the exponent randomly to simulate multiple flash fall off.
                if useAugmentation:
                    exponent = torch.exp(torch.randn((), mean=np.log(5), stddev=0.35))
                else:
                    exponent = 5.0
                lampFactor = lampFactor * torch.pow(distanceToConeCenter, exponent)
                print("using the distance to cone center")

        result = result * lampFactor
        result = result * torch.clamp(NdotL, min=0.0)

        if multiLight:
            result = (
                torch.sum(result, axis=2) * 1.0
            )  # if we have multiple light we need to multiply this by (1/number of lights).
        if lossRendering:
            result = result / torch.unsqueeze(
                torch.clamp(wiNorm[:, :, :, :, 2], min=0.001), axis=-1
            )  # This division is to compensate for the cosinus distribution of the intensity in the rendering.

        return [
            result
        ]  # , D_rendered, G_rendered, F_rendered, diffuse_rendered, diffuse]

    # generate the diffuse rendering for the loss computation
    def generateDiffuseRendering(self, batchSize, nbRenderings, targets, outputs):
        currentViewPos = generate_normalized_random_direction(
            batchSize, nbRenderings, lowEps=0.001, highEps=0.1
        )
        currentLightPos = generate_normalized_random_direction(
            batchSize, nbRenderings, lowEps=0.001, highEps=0.1
        )

        wi = currentLightPos
        wi = torch.unsqueeze(wi, axis=2)
        wi = torch.unsqueeze(wi, axis=2)

        wo = currentViewPos
        wo = torch.unsqueeze(wo, axis=2)
        wo = torch.unsqueeze(wo, axis=2)

        # Add a dimension to compensate for the nb of renderings
        # targets = tf.expand_dims(targets, axis=-2)
        # outputs = tf.expand_dims(outputs, axis=-2)

        wi = wi.to(targets.device)
        wo = wo.to(targets.device)

        # Here we have wi and wo with shape [batchSize, height,width, nbRenderings, 3]
        renderedDiffuse = self.render(
            targets, wi, wo, None, "diffuse", useAugmentation=False, lossRendering=True
        )[0]

        renderedDiffuseOutputs = self.render(
            outputs, wi, wo, None, "", useAugmentation=False, lossRendering=True
        )[
            0
        ]  # tf_Render_Optis(outputs,wi,wo)
        # renderedDiffuse = tf.Print(renderedDiffuse, [tf.shape(renderedDiffuse)],  message="This is renderings targets Diffuse: ", summarize=20)
        # renderedDiffuseOutputs = tf.Print(renderedDiffuseOutputs, [tf.shape(renderedDiffuseOutputs)],  message="This is renderings outputs Diffuse: ", summarize=20)
        return [renderedDiffuse, renderedDiffuseOutputs]

    # generate the specular rendering for the loss computation
    def generateSpecularRendering(
        self, batchSize, nbRenderings, surfaceArray, targets, outputs
    ):
        currentViewDir = generate_normalized_random_direction(
            batchSize, nbRenderings, lowEps=0.001, highEps=0.1
        )
        currentLightDir = currentViewDir * (
            torch.tensor((-1.0, -1.0, 1.0)).unsqueeze(0)
        )
        # Shift position to have highlight elsewhere than in the center.
        currentShift = torch.cat(
            [
                rand_range([batchSize, nbRenderings, 2], -1.0, 1.0),
                torch.zeros([batchSize, nbRenderings, 1], dtype=torch.float32) + 0.0001,
            ],
            axis=-1,
        )

        currentViewPos = (
            torch.mul(currentViewDir, generate_distance(batchSize, nbRenderings))
            + currentShift
        )
        currentLightPos = (
            torch.mul(currentLightDir, generate_distance(batchSize, nbRenderings))
            + currentShift
        )

        currentViewPos = torch.unsqueeze(currentViewPos, axis=2)
        currentViewPos = torch.unsqueeze(currentViewPos, axis=2)

        currentLightPos = torch.unsqueeze(currentLightPos, axis=2)
        currentLightPos = torch.unsqueeze(currentLightPos, axis=2)

        wo = currentViewPos - surfaceArray
        wi = currentLightPos - surfaceArray

        # targets = tf.expand_dims(targets, axis=-2)
        # outputs = tf.expand_dims(outputs, axis=-2)
        # targets = tf.Print(targets, [tf.shape(targets)],  message="This is targets in specu renderings: ", summarize=20)
        renderedSpecular = self.render(
            targets, wi, wo, None, "specu", useAugmentation=False, lossRendering=True
        )[0]
        renderedSpecularOutputs = self.render(
            outputs, wi, wo, None, "", useAugmentation=False, lossRendering=True
        )[0]
        # tf_Render_Optis(outputs,wi,wo, includeDiffuse = a.includeDiffuse)

        # renderedSpecularOutputs = tf.Print(renderedSpecularOutputs, [tf.shape(renderedSpecularOutputs)],  message="This is renderings outputs Specular: ", summarize=20)
        return [renderedSpecular, renderedSpecularOutputs]
    