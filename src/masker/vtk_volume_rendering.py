import vtk
from vtk.util.numpy_support import vtk_to_numpy # For converting vtk renders to numpy arrays
from PIL import Image # For saving/showing numpy arrays as images
import numpy as np

def getnumpyrender(fileName,viewup,azimuth,roll,voxeldims,minvoxel,maxvoxel):

    ## Set up colors       
    colors = vtk.vtkNamedColors()
    colors.SetColor("BkgColor", [255, 255, 255, 255]) # white background

    ## Create the renderer, the render window, interactor is currently commented out
    ren = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(ren)

    ## Set up reader; can be DICOM or NIFTI
    reader = vtk.vtkNIFTIImageReader() # NIFTI
    reader.SetFileName(fileName)

    ## The volume will be displayed by ray-cast alpha compositing.
    ## A ray-cast mapper is needed to do the ray-casting.
    volumeMapper = vtk.vtkFixedPointVolumeRayCastMapper()
    volumeMapper.SetInputConnection(reader.GetOutputPort())

    ## The color transfer function maps voxel intensities to colors; we want something very simple,
    ## so we make anything over 100 white
    volumeColor = vtk.vtkColorTransferFunction()
    volumeColor.AddRGBPoint(100, 1, 1, 1) 

    ## The opacity transfer function is used to control the opacity;
    ## it's a stepwise function so you need to set multiple values
    volumeScalarOpacity = vtk.vtkPiecewiseFunction()

    volumeScalarOpacity.AddPoint(minvoxel, 0.00)
    volumeScalarOpacity.AddPoint(maxvoxel, 1)
    
    ## The VolumeProperty attaches the color and opacity functions to the
    ## volume, and sets other volume properties
    volumeProperty = vtk.vtkVolumeProperty()
    volumeProperty.SetColor(volumeColor)
    volumeProperty.SetScalarOpacity(volumeScalarOpacity)
    volumeProperty.SetInterpolationTypeToLinear()
    volumeProperty.ShadeOn()
    volumeProperty.SetAmbient(0.8)
    volumeProperty.SetDiffuse(0.8)
    volumeProperty.SetSpecular(0.4)

    ## The vtkVolume is a vtkProp3D (like a vtkActor) and controls the position
    ## and orientation of the volume in world coordinates.
    volume = vtk.vtkVolume()
    volume.SetMapper(volumeMapper)
    volume.SetProperty(volumeProperty)

    ## Finally, add the volume to the renderer
    ren.AddViewProp(volume)

    ## Set up an initial view of the volume.  The focal point will be the
    ## center of the volume, and the camera position will be 400mm to the
    ## patient's left (which is our right).
    camera = ren.GetActiveCamera()
    c = volume.GetCenter()
    camera.SetViewUp(viewup) # this flips which way is up; this is what azimuth rotates around
    camera.SetPosition(c[0], c[1]*-1, c[2]) # the negative number MUST add up to the total Y dimension or there will be clipping! (e.g. y=256)
    camera.SetFocalPoint(c[0], c[1], c[2])
    camera.Azimuth(azimuth)
    camera.Elevation(0)
    camera.Roll(roll) # set camera roll according to arguments
    camera.ParallelProjectionOn() # Parallel projection is on ; what we want, but default is off

    ## Set a background color for the renderer
    ren.SetBackground(colors.GetColor3d("BkgColor"))

    ## Get dimensions of the volume; exacts dimensions is rounded integer values +1
    ## Set the size of the render window
    if(roll == 90 or roll == 270):
        renWin.SetSize(round(c[2]*2)+1, round(c[0]*2)+1) # Renders image at exact size of original mm dimensions
        camera.SetParallelScale(c[1])  ## Set this to same value as center; image fills entire render window
    else:
        renWin.SetSize(round(c[0]*2)+1, round(c[2]*2)+1) # Renders image at exact size of original mm dimensions
        camera.SetParallelScale(c[2])  ## Set this to same value as center; image fills entire render window

    ## Cast render to numpy array
    vtk_win_im = vtk.vtkWindowToImageFilter()
    vtk_win_im.SetInput(renWin)
    vtk_win_im.Update()

    vtk_image = vtk_win_im.GetOutput()

    width, height, _ = vtk_image.GetDimensions()

    vtk_array = vtk_image.GetPointData().GetScalars()
    components = vtk_array.GetNumberOfComponents()

    arr = vtk_to_numpy(vtk_array).reshape(height, width, components)

    ## Flip horizontally
    arr=np.flip(arr,1)

    return(arr)
    