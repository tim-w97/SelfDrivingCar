# Selbstfahrendes Auto

### Commands

Training starten

```shell
/opt/localdata/VirtualBoxVMs/ov/isaac_sim-2023.1.1/python.sh omniisaacgymenvs/scripts/rlgames_train.py task=SelfDrivingCar
```

<br/>

### Observations
- Vorschlag: 3 Lidar-Sensoren vorne am Auto

<br/>

### Actions

- Lenkachse 
  - -1 für links lenken und 
  - 1 für rechts lenken
- Beschleunigungsachse
  - -1 für Bremsen
  - 1 für Gas
