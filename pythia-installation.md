````markdown
## PYTHIA Particle Physics Installation Guide

**PYTHIA** is a general-purpose Monte Carlo event generator for high-energy physics collision events. It simulates interactions between electrons, protons, photons, and heavy nuclei, and is widely used in particle physics research for processes including hard and soft interactions, parton showers, and hadronization.[1]

### Overview of Installation Methods

There are multiple ways to install PYTHIA depending on your needs and computing environment. The main approaches include direct compilation from source, Python package installation, and integration with other frameworks like ROOT and HepMC.

### Basic Installation from Source (Linux/macOS)

The most straightforward installation method involves downloading and compiling PYTHIA directly:[1]

**Step 1: Download**  
Download the source tarball (currently version 8.316) from the official website to a suitable location.

**Step 2: Extract**  
Unzip and expand the archive:
```bash
tar xvfz pythia8316.tgz
````

**Step 3: Move to Directory**
Navigate to the newly created directory:

```bash
cd pythia8316
```

**Step 4: Configure (Optional)**
Read the `README` file for installation instructions. For basic installations without external library dependencies, you can skip this step. For advanced configurations (such as integrating ROOT, HepMC, or FastJet), configure with appropriate flags:[2]

```bash
./configure --prefix=$HOME/pythia8
```

For Python interface support:

```bash
./configure --with-python
```

**Step 5: Compile**
Compile the program:

```bash
make
```

This typically takes 1-3 minutes depending on your system. For parallel compilation on N cores:

```bash
make -jN
```

**Step 6: Test**
Move to the examples subdirectory and verify the installation by compiling and running a test example:

```bash
cd examples
make mainNNN
./mainNNN > mainNNN.log
```

where `NNN` is a three-digit number corresponding to an example program.[1]

### Python Installation Methods

For users preferring Python interfaces, PYTHIA can be installed via package managers:[2]

**Using pip:**

```bash
pip install pythia8mc
```

Note: The PyPI distribution uses the module name `pythia8mc` (not `pythia8`) due to naming availability.[2]

**Using conda:**

```bash
conda install -c conda-forge pythia8
```

This approach is particularly convenient when combining PYTHIA with other scientific tools. For example, to set up a complete environment with PYTHIA, HepMC2, and ROOT:[3]

```bash
conda create --name pythia8withhepmc2
conda activate pythia8withhepmc2
conda install -c conda-forge hepmc2 pythia8
conda install -c conda-forge root
```

### Installation with External Libraries

PYTHIA integrates with several external libraries that enhance functionality. Common ones include:

**HepMC (High Energy Physics Monte Carlo common event record)**
Used for standardized event output format:[4]

```bash
./configure --with-hepmc=path
```

**FastJet**
Jet clustering and analysis library, useful for analyzing generated events.

**ROOT**
Data analysis framework from CERN. During ROOT compilation, you can enable PYTHIA support:

```bash
./configure --enable-pythia8
```

**Compilation with Multiple Libraries**
When linking multiple external libraries during PYTHIA compilation:

```bash
./configure --with-hepmc=path --prefix=$HOME/pythia8
make
make install
```

After installation, set environment variables to make libraries discoverable:

```bash
export PYTHIA8=$HOME/pythia8
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PYTHIA8/lib
```

### System Requirements

PYTHIA 8 has minimal dependencies:[5]

**Compiler Requirements:**

* A C++11-compatible compiler (gcc ≥ 4.8 or clang ≥ 3.2)[6]

**Optional:**

* External libraries (HepMC, FastJet, ROOT) for enhanced functionality
* Python 3.x for Python interface

### Writing and Compiling Programs

PYTHIA programs are written in C++ using the Pythia class. Here's a minimal example structure:[7]

```cpp
#include "Pythia.h"
using namespace Pythia8;

int main() {
    Pythia pythia;
    pythia.readString("Beams:eCM = 7000");
    pythia.init(2212, 2212, 7000);
    
    for (int i = 0; i < 1000; ++i) {
        if (!pythia.next()) continue;
        // Process event
    }
    pythia.stat();
    return 0;
}
```

After creating your program (e.g., `myprog.cc`), compile it by editing the `Makefile` in the examples directory to include your program in the compilation targets, then type:

```bash
make myprog
./myprog > myprog.log
```

### Documentation and Resources

The PYTHIA 8.3 manual, available as arXiv:2203.11601, provides comprehensive documentation on physics models, usage, and configuration options. Key resources include:[8]

* Online HTML manual: Available in the unpacked source at `pythia8XXX/share/Pythia8/htmldoc/Welcome.html`
* PYTHIA 8 Worksheet for self-study and summer schools[4]
* Example programs in the `examples` subdirectory demonstrating common use cases
* Doxygen API documentation for quick reference of classes and methods

### Troubleshooting

For common issues during compilation:[9]

* Ensure your C++ compiler is properly installed and up-to-date
* Verify external library paths are correctly specified in configure
* Use separate build directories to avoid conflicts from previous compilation attempts
* Check that environment variables like `LD_LIBRARY_PATH` are properly set when linking external libraries

This flexible installation approach allows PYTHIA to be integrated into various computational workflows, from simple Monte Carlo simulations to complex high-energy physics analyses within frameworks like ROOT and HepMC.[3][1][2]

[1](https://pythia.org)
[2](https://pypi.org/project/pythia8mc/)
[3](https://a-kapoor.github.io/Pythia8_To_hepmc_To_ROOT/)
[4](https://pythia.org/download/pdf/worksheet8107.pdf)
[5](https://pythia.org/latest-manual/Welcome.html)
[6](https://theory.gsi.de/~smash/userguide/2.0/md_README.html)
[7](https://pythia.org/download/pdf/mergingworksheet8160.pdf)
[8](https://arxiv.org/abs/2203.11601)
[9](https://root-forum.cern.ch/t/problem-in-compiling-pythia8-code-with-root-link/21459)
[10](https://conway.physics.ucdavis.edu/teaching/245C/pages/pythia64.pdf)
[11](http://cp3.irmp.ucl.ac.be/~rouby/Pythia/index.html)
[12](https://theorique05.wordpress.com/master/semester-10/hep/pythia/)
[13](https://www.youtube.com/watch?v=FUNU_x2bXz4)
[14](https://aliceo2group.github.io/docs/d3/d68/refrunSimExamplesPythia.html)
[15](https://indico.fjfi.cvut.cz/event/268/contributions/4451/attachments/1678/3546/PYTHIA_Mezhenska-1.pdf)
[16](https://github.com/SHHam12/Root-Cern)
[17](https://hep-fcc.github.io/fcc-tutorials/main/fast-sim-and-analysis/FccFastSimGeneration.html)
[18](https://www.nevis.columbia.edu/~haas/documents/pythia6400.pdf)
[19](http://www.graverini.net/elena/computing/physics-software/install-pythia/)
[20](https://github.com/JeffersonLab/claspyth)
[21](https://cds.cern.ch/record/2296395/files/pythia.pdf)
[22](https://indico.cern.ch/event/669093/attachments/1615913/2770286/TutorialSchoolPisav2.pdf)
[23](https://www.hep.phy.cam.ac.uk/theory/webber/MCnet/MClecture4.pdf)
[24](https://www.youtube.com/watch?v=1Wtq8MGgxHk)
[25](https://skands.physics.monash.edu/slides/pdf/12-tools-skands.pdf)
[26](https://www.pythea.org/en/docs/installing.html)
[27](https://root.cern.ch/d/build-root-old-method.html)
[28](https://pypi.org/project/pythia-uq/2.0.0/)
[29](https://foundations.projectpythia.org/foundations/conda/)
[30](https://www.youtube.com/watch?v=QhwNuEKty3Y)
[31](https://foundations.projectpythia.org/foundations/how-to-run-python)
[32](https://answers.launchpad.net/mg5amcnlo/+question/632807)

```
```
