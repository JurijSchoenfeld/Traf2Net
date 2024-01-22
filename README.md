# Traf2Net

## Micro Level Contact Networks from Macro Level Activity Trajectories

### Overview

This project focuses on generating micro-level contact networks from macro-level activity trajectories obtained through TAPAS, a sophisticated travel demand model. The micro-level contact networks are created using various human mobility models and simpler generative dynamic network models. The generated networks are then compared to real-world empirical contact networks for evaluation.

### Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Data](#data)
5. [Models](#models)
6. [Evaluation](#evaluation)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)
10. [Contact](#contact)

### Introduction

In the context of the Master Thesis, this project aims to bridge the gap between macro-level activity trajectories and micro-level contact networks. TAPAS, a highly advanced travel demand model, provides the macro-level activity trajectories that serve as the foundation for generating micro-level contact networks. The project explores different human mobility models and simpler generative dynamic network models for this purpose.

### Installation

To run this project, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/your-project.git

### Data

The data used in this project is obtained from TAPAS, a sophisticated travel demand model. Unfortunately, the TAPAS data is not publicly available due to license issues and its large file size. However, it may be provided upon request.

To acquire the TAPAS data for replication or further analysis, please contact me to request access. Upon approval, the data can be shared securely based on the agreed-upon terms and conditions.

Additionally, if there are any specific preprocessing steps applied to the TAPAS data before using it in this project, those details will be provided in the relevant sections of the codebase or documentation.

### Models

This project employs various models to generate micro-level contact networks based on the macro-level activity trajectories provided by TAPAS. Trajectories activate and deactivate nodes at specific locations, and links between active nodes are established using on of the following models:

**Simple Generative Dynamic Network Models:**

1. **Baseline Approach:**
   - Description: A fully connected dynamic network is created between all active nodes.
   - Implementation: [contac_networks.ContactNetworks.baseline]

2. **Random Approach:**
   - Description: Contacts from the baseline approach are randomly chosen, and the contact duration is drawn from a given probability distribution.
   - Implementation: [contac_networks.ContactNetworks.random]

3. **Clique Approach:**
   - Description: Locations are divided into spaces, and nodes are assigned to default spaces. Nodes leave the default space with a given probability, and the duration of their stay at a different location is drawn from a given distribution. All nodes in a certain space are fully connected.
   - Implementation: [contac_networks.ContactNetworks.clique]

**Sophisticated Human Mobility Models:**

4. **Random Waypoint Model (RWP):**
   - Description: Movement of nodes is simulated using a random walk with pauses. Distances between active nodes are calculated at every time step, and a contact is generated if the distance surpasses a given threshold.
   - Implementation: [mobility.py]

5. **Truncated Levy Walk (TLW):**
   - Description: Similar to RWP but with added Levy flights during pauses.
   - Implementation: [mobility.py]

6. **Spatio Temporal Parametric Stepping Model (STEPS):**
   - Description: Locations are divided into spaces, and each node is assigned a default space. Nodes leave their default space with a given probability, and the new space is chosen based on a probability distribution determined by the distance to the default space.
   - Implementation: [contac_networks.ContactNetworks.STEPS]

7. **STEPS + RWP:**
   - Description: Nodes swap spaces according to STEPS movement, and the movement within the space is simulated with RWP.
   - Implementation: [contac_networks.ContactNetworks.STEPS_with_RWP]

More models might be integrated in the future.


### Evaluation

To assess the performance and quality of the generated micro-level contact networks, the evaluation process involves utilizing empirical networks from various scenarios, including a supermarket, an office, and a school. Diverse empirical settings provide a robust foundation for comprehensive evaluation. The evaluation metrics encompass various aspects, including contact duration, inter-contact time, and network density, to gauge the accuracy and realism of the generated networks. Special attention is given to higher-order network properties such as communities and clustering, providing insights into the intricate structure of the micro-level networks. Additionally, the infection dynamics within these networks are analyzed using an Agent-Based SIR (Susceptible-Infectious-Recovered) model.

#### Agent-Based SIR Model:

The Agent-Based SIR model is employed to simulate the spread of infections within the generated networks. In this model, each individual (or agent) in the network is classified into one of three compartments: Susceptible (S), Infectious (I), or Recovered (R). The model's dynamics are governed by the following set of equations:

- **Susceptible to Infectious Transition:**
  
  $$
  \frac{dS}{dt} = -\beta \frac{SI}{N}
  $$

- **Infectious to Recovered Transitions:**
    $$
    \frac{dI}{dt} = \beta \frac{SI}{N} - \gamma I
    $$

- **Recovered individuals remain constant over time:**
    $$
    \frac{dI}{dt} = \beta \frac{SI}{N} - \gamma I
    $$

Where:
- **S** is the number of susceptible individuals,
- **I** is the number of infectious individuals,
- **R** is the number of recovered individuals,
- **β** is the transmission rate,
- **γ** is the recovery rate,
- **N** is the total population.

The Agent-Based SIR model allows for a dynamic simulation of infection spread within the generated micro-level contact networks, facilitating a comparative analysis of infection dynamics just above the epidemic threshold.

New metrics are currently being tested to further enhance the evaluation process. As the project progresses, additional metrics may be introduced to provide a more comprehensive and nuanced assessment of the generated micro-level contact networks.


