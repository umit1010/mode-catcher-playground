# mode-catcher-playground
 
`mode-catcher` is an experimental project to see if we can use simple NLP tools and network analysis methods to develop conceptual models of thinking/reasoning patterns from my dissertation interview transcripts.

My dissertation study focuses on a new research interview protocol. I sit with an adult layperson and help them create a computational model for their chosen real-world topic. They describe their model to me, and I convert their ideas into NetLogo code. I am currently analyzing these interview recordings to find out (1) if there are any salient patterns in how the participants built their models and (2) if we can use this data to uncover how laypeople reason about real-world complex systems and emergent phenomena.

I use network analysis methods because my dissertation study builds on a modern theory of knowledge called Knowledge-in-Pieces (KiP; diSessa, 1993). KiP conceptualizes human knowledge as a complex heterogenous network of primitive elements + argues that human reasoning is a spontaneous process of activation and restructuration of the underlying knowledge network.

I use simple statistics-based NLP tools in this project (instead of LLMs) because I don’t want the code to spit out a model of a participant’s reasoning. I want to create the model myself, but I want to develop an app that makes the process easier.

*Note: This project is a proof-of-concept prototype, and I am still actively working on it.*


## Relevant literature

You can read the following three short papers to get started with the underlying theoretical ideas of the project.

* Sherin et al. (2007) Conceptual Dynamics in Clinical Interviews https://doi.org/10.1063/1.2820937
* Wilensky (2001) Modeling Nature’s Emergent Patterns with Multi-Agent Languages https://ccl.northwestern.edu/2013/mnep9.pdf
* Moretti (2011) Network Theory, Plot Analysis https://litlab.stanford.edu/projects/network-theory-plot-analysis/


## Setting up your Python development environment

I recommend using a conda-based setup to get the current version of the playground up and running on your computer. I prepared a list of instructions below that should work fine on Mac or Linux, but if you use a Windows computer or run into any issues, let me know, and we can troubleshoot via a Zoom meeting:

* If you have not done so already, install `conda`: 
  * Here is a tutorial: https://docs.anaconda.com/free/miniconda/miniconda-install/ 
  * Once conda is installed, open a new terminal window, and you should see the prompt change to `(base) ~`

* Navigate to a folder of your choosing and clone the git repository:
  * `git clone https://github.com/umit1010/mode-catcher-playground.git`

* Navigate to the newly created `mode-catcher-playground` directory
  * `cd mode-catcher-playground`

* Create a new virtual environment using the following command in the terminal:
  * `conda env create -f environment.yml`

* Once the new environment is created, activate it:
  * `conda activate mode-catcher-main`

* Run the server:
  * `python app.py`

If everything went as expected, you should see the message “Dash is running on http://127.0.0.1:8050/.” 

To use the app, navigate to the link http://127.0.0.1:8050/.
