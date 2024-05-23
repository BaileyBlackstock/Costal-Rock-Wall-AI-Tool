# Costal-Rock-Wall-AI-Tool
This projects aim is to take a photo of rock armour with a hard hat in the picture, and have a tool that then estimates the rock sizes in the image and produces a rock size distribution.

## Steps to converting the .py file into a .exe file
1. open terminal.
2. run "pip install pyinstaller".
3. run "pyinstaller --onefile AiImageAnalysis.py".
4. find the .exe file in the new folder 'dist'.

## Instructions
This program works by taking a silhouette of a rock wall with the rocks and reference image \(a standard sie hard hat) filled in and produces outputs in the forms of a csv file and graphs via the python matplotlib library. The first step for this program is to create the silhouette input image, make sure to do this using the following rules:
* The hard hat must be ecoloured white and the only thing coloured white.
* The background must be black, ensure there is no rim of colour at the edges.
* The rocks must be moderately distinct colours, i.e. no two shades of red.
* The rocks can be the same colour but must not touch.
* Although not rules, it helps to only colour in rocks on a similar depth as the hard hat for the best results, and to only use 4 colours for the rocks \(you will never need more than this).

Once this image is created use this program by executing the following steps in order:
1. Input image/s in the input folder.
2. Edit the settings.txt file with the correct data, the values after "rock_colours_per_images" should be the number of different colours on rocks in each image alphabeticlly.
3. Run AiImageAnalysis.exe.

The results will then be displayed in a percentage graph and put into a newly created file called outputs.cvs.
