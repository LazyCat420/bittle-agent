export const BUILTIN_MOVESETS = {
  "sit": {
    "name": "sit",
    "label": "Sit",
    "description": "Standard Bittle seated posture",
    "kind": "posture",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": -30,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": 40,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 8,
        "delay_ms": 300
      }
    ]
  },
  "balance": {
    "name": "balance",
    "label": "Stand / Balance",
    "description": "Neutral four-legged standing posture",
    "kind": "posture",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 300
      }
    ]
  },
  "rest": {
    "name": "rest",
    "label": "Rest (Relax)",
    "description": "Flat rested posture on belly",
    "kind": "posture",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -55,
          "9": -55,
          "10": 55,
          "11": 55,
          "12": 60,
          "13": 60,
          "14": 60,
          "15": 60
        },
        "speed_deg_per_step": 8,
        "delay_ms": 300
      }
    ]
  },
  "ck": {
    "name": "ck",
    "label": "Check Around",
    "description": "Scans left and right with head pan while standing steady",
    "kind": "behavior",
    "frames": [
      {
        "angles": {
          "0": 45,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 6,
        "delay_ms": 350
      },
      {
        "angles": {
          "0": -45,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 6,
        "delay_ms": 350
      },
      {
        "angles": {
          "0": 30,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 6,
        "delay_ms": 250
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 6,
        "delay_ms": 200
      }
    ]
  },
  "hi": {
    "name": "hi",
    "label": "Wave Hello",
    "description": "Sits down and waves front-right paw twice",
    "kind": "behavior",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": -30,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": 40,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      },
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": 20,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": -25,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 8,
        "delay_ms": 200
      },
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": 35,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": 15,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 12,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": 15,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": -30,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 12,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": 35,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": 15,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 12,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": 20,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": -25,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 8,
        "delay_ms": 200
      },
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": -30,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": 40,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      }
    ]
  },
  "pu": {
    "name": "pu",
    "label": "Push Ups",
    "description": "Lowers front chest to floor and presses up two times",
    "kind": "behavior",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -75,
          "9": -75,
          "10": -45,
          "11": -45,
          "12": 105,
          "13": 105,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 6,
        "delay_ms": 350
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 300
      },
      {
        "angles": {
          "0": 0,
          "8": -75,
          "9": -75,
          "10": -45,
          "11": -45,
          "12": 105,
          "13": 105,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 6,
        "delay_ms": 350
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 300
      }
    ]
  },
  "nd": {
    "name": "nd",
    "label": "Nod",
    "description": "Affirmative nodding motion",
    "kind": "behavior",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -55,
          "9": -55,
          "10": -35,
          "11": -35,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 200
      },
      {
        "angles": {
          "0": 0,
          "8": -35,
          "9": -35,
          "10": -55,
          "11": -55,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 200
      },
      {
        "angles": {
          "0": 0,
          "8": -55,
          "9": -55,
          "10": -35,
          "11": -35,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 200
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 200
      }
    ]
  },
  "bf": {
    "name": "bf",
    "label": "Backflip",
    "description": "High-energy backward somersault landing squarely on all four feet",
    "kind": "behavior",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -80,
          "9": -80,
          "10": 75,
          "11": 75,
          "12": 110,
          "13": 110,
          "14": 70,
          "15": 70
        },
        "speed_deg_per_step": 12,
        "delay_ms": 250
      },
      {
        "angles": {
          "0": 0,
          "8": 30,
          "9": 30,
          "10": -90,
          "11": -90,
          "12": 20,
          "13": 20,
          "14": 120,
          "15": 120
        },
        "speed_deg_per_step": 30,
        "delay_ms": 150
      },
      {
        "angles": {
          "0": 0,
          "8": -70,
          "9": -70,
          "10": 70,
          "11": 70,
          "12": 90,
          "13": 90,
          "14": 90,
          "15": 90
        },
        "speed_deg_per_step": 30,
        "delay_ms": 200
      },
      {
        "angles": {
          "0": 0,
          "8": -60,
          "9": -60,
          "10": -60,
          "11": -60,
          "12": 95,
          "13": 95,
          "14": 95,
          "15": 95
        },
        "speed_deg_per_step": 16,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      }
    ]
  },
  "pee": {
    "name": "pee",
    "label": "Pee",
    "description": "Playful single rear-leg lift posture",
    "kind": "behavior",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -30,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 40,
          "15": 80
        },
        "speed_deg_per_step": 6,
        "delay_ms": 800
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 6,
        "delay_ms": 300
      }
    ]
  },
  "str": {
    "name": "str",
    "label": "Stretch",
    "description": "Full body waking stretch",
    "kind": "behavior",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -75,
          "9": -75,
          "10": -20,
          "11": -20,
          "12": 110,
          "13": 110,
          "14": 50,
          "15": 50
        },
        "speed_deg_per_step": 5,
        "delay_ms": 600
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 6,
        "delay_ms": 300
      }
    ]
  },
  "zero": {
    "name": "zero",
    "label": "Zero / Calibration Pose",
    "description": "All servo motor angles set to zero",
    "kind": "posture",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": 0,
          "9": 0,
          "10": 0,
          "11": 0,
          "12": 0,
          "13": 0,
          "14": 0,
          "15": 0
        },
        "speed_deg_per_step": 8,
        "delay_ms": 300
      }
    ]
  },
  "up": {
    "name": "up",
    "label": "Stand Up",
    "description": "Stands up to standard neutral balance posture",
    "kind": "posture",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 300
      }
    ]
  },
  "calib": {
    "name": "calib",
    "label": "Calibration Posture",
    "description": "Orthogonal servo alignment check",
    "kind": "posture",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": 0,
          "9": 0,
          "10": 0,
          "11": 0,
          "12": 0,
          "13": 0,
          "14": 0,
          "15": 0
        },
        "speed_deg_per_step": 8,
        "delay_ms": 300
      }
    ]
  },
  "wkF": {
    "name": "wkF",
    "label": "Walk Forward",
    "description": "Authentic diagonal walking stride across the floor",
    "kind": "gait",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -65,
          "9": -30,
          "10": -55,
          "11": -25,
          "12": 95,
          "13": 70,
          "14": 85,
          "15": 65
        },
        "speed_deg_per_step": 10,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 10,
        "delay_ms": 140
      },
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": -65,
          "10": -25,
          "11": -55,
          "12": 70,
          "13": 95,
          "14": 65,
          "15": 85
        },
        "speed_deg_per_step": 10,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 10,
        "delay_ms": 140
      }
    ]
  },
  "wkL": {
    "name": "wkL",
    "label": "Walk Left",
    "description": "Turning walk to the left",
    "kind": "gait",
    "frames": [
      {
        "angles": {
          "0": 25,
          "8": -70,
          "9": -45,
          "10": -30,
          "11": -45,
          "12": 95,
          "13": 80,
          "14": 70,
          "15": 80
        },
        "speed_deg_per_step": 10,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 10,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 10,
        "delay_ms": 140
      },
      {
        "angles": {
          "0": 25,
          "8": -45,
          "9": -50,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 75,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 10,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 10,
        "delay_ms": 140
      }
    ]
  },
  "wkR": {
    "name": "wkR",
    "label": "Walk Right",
    "description": "Turning walk to the right",
    "kind": "gait",
    "frames": [
      {
        "angles": {
          "0": -25,
          "8": -45,
          "9": -70,
          "10": -45,
          "11": -30,
          "12": 80,
          "13": 95,
          "14": 80,
          "15": 70
        },
        "speed_deg_per_step": 10,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": -10,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 10,
        "delay_ms": 140
      },
      {
        "angles": {
          "0": -25,
          "8": -50,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 75,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 10,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 10,
        "delay_ms": 140
      }
    ]
  },
  "trF": {
    "name": "trF",
    "label": "Trot Forward",
    "description": "High-cadence diagonal trotting gait",
    "kind": "gait",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -70,
          "9": -45,
          "10": -45,
          "11": -20,
          "12": 100,
          "13": 80,
          "14": 80,
          "15": 60
        },
        "speed_deg_per_step": 14,
        "delay_ms": 140
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 14,
        "delay_ms": 100
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -70,
          "10": -20,
          "11": -45,
          "12": 80,
          "13": 100,
          "14": 60,
          "15": 80
        },
        "speed_deg_per_step": 14,
        "delay_ms": 140
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 14,
        "delay_ms": 100
      }
    ]
  },
  "crF": {
    "name": "crF",
    "label": "Crawl Forward",
    "description": "Low-profile creeping crawl",
    "kind": "gait",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -70,
          "9": -30,
          "10": -60,
          "11": -25,
          "12": 105,
          "13": 70,
          "14": 95,
          "15": 65
        },
        "speed_deg_per_step": 6,
        "delay_ms": 220
      },
      {
        "angles": {
          "0": 0,
          "8": -55,
          "9": -55,
          "10": -45,
          "11": -45,
          "12": 90,
          "13": 90,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 6,
        "delay_ms": 160
      },
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": -70,
          "10": -25,
          "11": -60,
          "12": 70,
          "13": 105,
          "14": 65,
          "15": 95
        },
        "speed_deg_per_step": 6,
        "delay_ms": 220
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 6,
        "delay_ms": 160
      }
    ]
  },
  "bk": {
    "name": "bk",
    "label": "Back Up",
    "description": "Reverse stepping motion",
    "kind": "gait",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -25,
          "9": -45,
          "10": -45,
          "11": -65,
          "12": 65,
          "13": 80,
          "14": 80,
          "15": 95
        },
        "speed_deg_per_step": 8,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 140
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -25,
          "10": -65,
          "11": -45,
          "12": 80,
          "13": 65,
          "14": 95,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 140
      }
    ]
  },
  "fiv": {
    "name": "fiv",
    "label": "High Five",
    "description": "Sits and raises front paw for a high five",
    "kind": "behavior",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": -30,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": 40,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      },
      {
        "angles": {
          "0": 0,
          "8": 45,
          "9": -30,
          "10": 80,
          "11": 80,
          "12": 20,
          "13": 40,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 10,
        "delay_ms": 700
      },
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": -30,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": 40,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      }
    ]
  },
  "gdb": {
    "name": "gdb",
    "label": "Good Boy",
    "description": "Affectionate happy head wobble and praise posture",
    "kind": "behavior",
    "frames": [
      {
        "angles": {
          "0": 30,
          "8": -40,
          "9": -50,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 10,
        "delay_ms": 200
      },
      {
        "angles": {
          "0": -30,
          "8": -50,
          "9": -40,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 10,
        "delay_ms": 200
      },
      {
        "angles": {
          "0": 20,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 10,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 200
      }
    ]
  },
  "hsk": {
    "name": "hsk",
    "label": "Handshake",
    "description": "Sits down and extends front right paw for a handshake",
    "kind": "behavior",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": -30,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": 40,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      },
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": 25,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": -10,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 8,
        "delay_ms": 600
      },
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": 35,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": 0,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 10,
        "delay_ms": 200
      },
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": 20,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": -15,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 10,
        "delay_ms": 200
      },
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": -30,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": 40,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      }
    ]
  },
  "hu": {
    "name": "hu",
    "label": "Hands Up",
    "description": "Rears onto hind legs with both front paws raised",
    "kind": "behavior",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": -30,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": 40,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      },
      {
        "angles": {
          "0": 0,
          "8": 45,
          "9": 45,
          "10": 80,
          "11": 80,
          "12": 15,
          "13": 15,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 10,
        "delay_ms": 700
      },
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": -30,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": 40,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      }
    ]
  },
  "jmp": {
    "name": "jmp",
    "label": "Jump",
    "description": "Crouches and leaps forward",
    "kind": "behavior",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -75,
          "9": -75,
          "10": -75,
          "11": -75,
          "12": 105,
          "13": 105,
          "14": 105,
          "15": 105
        },
        "speed_deg_per_step": 12,
        "delay_ms": 250
      },
      {
        "angles": {
          "0": 0,
          "8": -15,
          "9": -15,
          "10": -15,
          "11": -15,
          "12": 45,
          "13": 45,
          "14": 45,
          "15": 45
        },
        "speed_deg_per_step": 26,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 12,
        "delay_ms": 250
      }
    ]
  },
  "kc": {
    "name": "kc",
    "label": "Kick",
    "description": "Forward snapping front-leg kick",
    "kind": "behavior",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -20,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 50,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 10,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": 40,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": -30,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 24,
        "delay_ms": 220
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 12,
        "delay_ms": 200
      }
    ]
  },
  "rc": {
    "name": "rc",
    "label": "Recover",
    "description": "Righting maneuver to restore stable standing balance",
    "kind": "behavior",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -55,
          "9": -55,
          "10": 55,
          "11": 55,
          "12": 60,
          "13": 60,
          "14": 60,
          "15": 60
        },
        "speed_deg_per_step": 8,
        "delay_ms": 200
      },
      {
        "angles": {
          "0": 0,
          "8": -70,
          "9": -70,
          "10": 55,
          "11": 55,
          "12": 95,
          "13": 95,
          "14": 60,
          "15": 60
        },
        "speed_deg_per_step": 10,
        "delay_ms": 200
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      }
    ]
  },
  "zz": {
    "name": "zz",
    "label": "Sleep",
    "description": "Lies down comfortably to sleep",
    "kind": "behavior",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -55,
          "9": -55,
          "10": 55,
          "11": 55,
          "12": 60,
          "13": 60,
          "14": 60,
          "15": 60
        },
        "speed_deg_per_step": 6,
        "delay_ms": 800
      }
    ]
  },
  "snf": {
    "name": "snf",
    "label": "Sniff",
    "description": "Lowers head to sniff the ground left and right",
    "kind": "behavior",
    "frames": [
      {
        "angles": {
          "0": 25,
          "8": -60,
          "9": -60,
          "10": -45,
          "11": -45,
          "12": 95,
          "13": 95,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 6,
        "delay_ms": 300
      },
      {
        "angles": {
          "0": -25,
          "8": -60,
          "9": -60,
          "10": -45,
          "11": -45,
          "12": 95,
          "13": 95,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 6,
        "delay_ms": 300
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      }
    ]
  },
  "hds": {
    "name": "hds",
    "label": "Handstand",
    "description": "Elevates rear legs into a handstand on front paws",
    "kind": "behavior",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -60,
          "9": -60,
          "10": -85,
          "11": -85,
          "12": 80,
          "13": 80,
          "14": 110,
          "15": 110
        },
        "speed_deg_per_step": 8,
        "delay_ms": 700
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 300
      }
    ]
  },
  "chr": {
    "name": "chr",
    "label": "Cheer",
    "description": "Excited cheering with alternating front paws",
    "kind": "behavior",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": -30,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": 40,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 8,
        "delay_ms": 200
      },
      {
        "angles": {
          "0": 0,
          "8": 35,
          "9": -20,
          "10": 80,
          "11": 80,
          "12": 15,
          "13": 40,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 14,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": -20,
          "9": 35,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": 15,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 14,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": 35,
          "9": -20,
          "10": 80,
          "11": 80,
          "12": 15,
          "13": 40,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 14,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      }
    ]
  },
  "dg": {
    "name": "dg",
    "label": "Dig",
    "description": "Front paws alternate digging the ground",
    "kind": "behavior",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -60,
          "9": -20,
          "10": -45,
          "11": -45,
          "12": 100,
          "13": 50,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 12,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": -20,
          "9": -60,
          "10": -45,
          "11": -45,
          "12": 50,
          "13": 100,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 12,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": -60,
          "9": -20,
          "10": -45,
          "11": -45,
          "12": 100,
          "13": 50,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 12,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 200
      }
    ]
  },
  "pd": {
    "name": "pd",
    "label": "Play Dead",
    "description": "Drops completely limp onto floor",
    "kind": "behavior",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -55,
          "9": -55,
          "10": 55,
          "11": 55,
          "12": 60,
          "13": 60,
          "14": 60,
          "15": 60
        },
        "speed_deg_per_step": 10,
        "delay_ms": 800
      }
    ]
  },
  "ang": {
    "name": "ang",
    "label": "Angry",
    "description": "Aggressive posture with head tilted down and chest low",
    "kind": "behavior",
    "frames": [
      {
        "angles": {
          "0": 20,
          "8": -65,
          "9": -65,
          "10": -45,
          "11": -45,
          "12": 100,
          "13": 100,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 12,
        "delay_ms": 300
      },
      {
        "angles": {
          "0": -20,
          "8": -65,
          "9": -65,
          "10": -45,
          "11": -45,
          "12": 100,
          "13": 100,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 12,
        "delay_ms": 300
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      }
    ]
  },
  "lnd": {
    "name": "lnd",
    "label": "Landing Pose",
    "description": "Lifted legs posture ready for surface touch-down",
    "kind": "posture",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": -30,
          "10": -30,
          "11": -30,
          "12": 60,
          "13": 60,
          "14": 60,
          "15": 60
        },
        "speed_deg_per_step": 8,
        "delay_ms": 300
      }
    ]
  },
  "trL": {
    "name": "trL",
    "label": "Trot Left",
    "description": "Trotting gait turning left",
    "kind": "gait",
    "frames": [
      {
        "angles": {
          "0": 25,
          "8": -70,
          "9": -45,
          "10": -45,
          "11": -20,
          "12": 100,
          "13": 80,
          "14": 80,
          "15": 60
        },
        "speed_deg_per_step": 14,
        "delay_ms": 140
      },
      {
        "angles": {
          "0": 10,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 14,
        "delay_ms": 100
      },
      {
        "angles": {
          "0": 25,
          "8": -45,
          "9": -55,
          "10": -30,
          "11": -45,
          "12": 80,
          "13": 85,
          "14": 70,
          "15": 80
        },
        "speed_deg_per_step": 14,
        "delay_ms": 140
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 14,
        "delay_ms": 100
      }
    ]
  },
  "crL": {
    "name": "crL",
    "label": "Crawl Left",
    "description": "Low crawl turning left",
    "kind": "gait",
    "frames": [
      {
        "angles": {
          "0": 25,
          "8": -70,
          "9": -45,
          "10": -60,
          "11": -45,
          "12": 105,
          "13": 80,
          "14": 95,
          "15": 80
        },
        "speed_deg_per_step": 6,
        "delay_ms": 220
      },
      {
        "angles": {
          "0": 10,
          "8": -55,
          "9": -55,
          "10": -45,
          "11": -45,
          "12": 90,
          "13": 90,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 6,
        "delay_ms": 160
      },
      {
        "angles": {
          "0": 25,
          "8": -45,
          "9": -50,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 90,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 6,
        "delay_ms": 220
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 6,
        "delay_ms": 160
      }
    ]
  },
  "vtF": {
    "name": "vtF",
    "label": "Step In Place",
    "description": "Stepping in place forward cadence",
    "kind": "gait",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -65,
          "9": -45,
          "10": -45,
          "11": -25,
          "12": 95,
          "13": 80,
          "14": 80,
          "15": 65
        },
        "speed_deg_per_step": 10,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 10,
        "delay_ms": 140
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -65,
          "10": -25,
          "11": -45,
          "12": 80,
          "13": 95,
          "14": 65,
          "15": 80
        },
        "speed_deg_per_step": 10,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 10,
        "delay_ms": 140
      }
    ]
  },
  "vtL": {
    "name": "vtL",
    "label": "Step In Place Left",
    "description": "Stepping in place turning left",
    "kind": "gait",
    "frames": [
      {
        "angles": {
          "0": 30,
          "8": -65,
          "9": -45,
          "10": -25,
          "11": -45,
          "12": 95,
          "13": 80,
          "14": 65,
          "15": 80
        },
        "speed_deg_per_step": 10,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 15,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 10,
        "delay_ms": 140
      },
      {
        "angles": {
          "0": 30,
          "8": -45,
          "9": -55,
          "10": -45,
          "11": -35,
          "12": 80,
          "13": 85,
          "14": 80,
          "15": 75
        },
        "speed_deg_per_step": 10,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 10,
        "delay_ms": 140
      }
    ]
  },
  "bkL": {
    "name": "bkL",
    "label": "Back Up Left",
    "description": "Reversing while turning left",
    "kind": "gait",
    "frames": [
      {
        "angles": {
          "0": 25,
          "8": -25,
          "9": -45,
          "10": -45,
          "11": -65,
          "12": 65,
          "13": 80,
          "14": 80,
          "15": 95
        },
        "speed_deg_per_step": 8,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 10,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 140
      },
      {
        "angles": {
          "0": 25,
          "8": -45,
          "9": -35,
          "10": -55,
          "11": -45,
          "12": 80,
          "13": 75,
          "14": 85,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 140
      }
    ]
  },
  "phF": {
    "name": "phF",
    "label": "Push-up Walk Forward",
    "description": "Low chest crawl walk",
    "kind": "gait",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -75,
          "9": -75,
          "10": -40,
          "11": -45,
          "12": 110,
          "13": 110,
          "14": 75,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 200
      },
      {
        "angles": {
          "0": 0,
          "8": -70,
          "9": -70,
          "10": -45,
          "11": -40,
          "12": 105,
          "13": 105,
          "14": 80,
          "15": 75
        },
        "speed_deg_per_step": 8,
        "delay_ms": 200
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 200
      }
    ]
  },
  "phL": {
    "name": "phL",
    "label": "Push-up Walk Left",
    "description": "Low chest crawl turning left",
    "kind": "gait",
    "frames": [
      {
        "angles": {
          "0": 25,
          "8": -75,
          "9": -75,
          "10": -40,
          "11": -45,
          "12": 110,
          "13": 110,
          "14": 75,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 200
      },
      {
        "angles": {
          "0": 10,
          "8": -70,
          "9": -70,
          "10": -45,
          "11": -40,
          "12": 105,
          "13": 105,
          "14": 80,
          "15": 75
        },
        "speed_deg_per_step": 8,
        "delay_ms": 200
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 200
      }
    ]
  },
  "bdF": {
    "name": "bdF",
    "label": "Bound Forward",
    "description": "Energetic bound leap stride",
    "kind": "gait",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -80,
          "9": -80,
          "10": -80,
          "11": -80,
          "12": 110,
          "13": 110,
          "14": 110,
          "15": 110
        },
        "speed_deg_per_step": 12,
        "delay_ms": 220
      },
      {
        "angles": {
          "0": 0,
          "8": 10,
          "9": 10,
          "10": -20,
          "11": -20,
          "12": 30,
          "13": 30,
          "14": 50,
          "15": 50
        },
        "speed_deg_per_step": 24,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 10,
        "delay_ms": 220
      }
    ]
  },
  "jpF": {
    "name": "jpF",
    "label": "Jump Forward",
    "description": "Crouching jump stride forward",
    "kind": "gait",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -75,
          "9": -75,
          "10": -75,
          "11": -75,
          "12": 105,
          "13": 105,
          "14": 105,
          "15": 105
        },
        "speed_deg_per_step": 12,
        "delay_ms": 240
      },
      {
        "angles": {
          "0": 0,
          "8": -10,
          "9": -10,
          "10": -10,
          "11": -10,
          "12": 40,
          "13": 40,
          "14": 40,
          "15": 40
        },
        "speed_deg_per_step": 26,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 12,
        "delay_ms": 240
      }
    ]
  },
  "ff": {
    "name": "ff",
    "label": "Fart / Play",
    "description": "Playful butt-raise wiggle",
    "kind": "behavior",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -70,
          "9": -70,
          "10": -15,
          "11": -15,
          "12": 100,
          "13": 100,
          "14": 50,
          "15": 50
        },
        "speed_deg_per_step": 8,
        "delay_ms": 350
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -30,
          "11": -60,
          "12": 80,
          "13": 80,
          "14": 65,
          "15": 95
        },
        "speed_deg_per_step": 14,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -60,
          "11": -30,
          "12": 80,
          "13": 80,
          "14": 95,
          "15": 65
        },
        "speed_deg_per_step": 14,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      }
    ]
  },
  "hg": {
    "name": "hg",
    "label": "Hug",
    "description": "Sits and opens front paws wide",
    "kind": "behavior",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": -30,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": 40,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      },
      {
        "angles": {
          "0": 0,
          "8": 40,
          "9": 40,
          "10": 80,
          "11": 80,
          "12": 20,
          "13": 20,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 10,
        "delay_ms": 600
      },
      {
        "angles": {
          "0": 0,
          "8": 50,
          "9": 50,
          "10": 80,
          "11": 80,
          "12": 0,
          "13": 0,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 10,
        "delay_ms": 400
      },
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": -30,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": 40,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      }
    ]
  },
  "mw": {
    "name": "mw",
    "label": "Meow / Sound",
    "description": "Tilts head up and chirps",
    "kind": "behavior",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -35,
          "9": -35,
          "10": -45,
          "11": -45,
          "12": 70,
          "13": 70,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 6,
        "delay_ms": 300
      },
      {
        "angles": {
          "0": 20,
          "8": -35,
          "9": -35,
          "10": -45,
          "11": -45,
          "12": 70,
          "13": 70,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 300
      },
      {
        "angles": {
          "0": -20,
          "8": -35,
          "9": -35,
          "10": -45,
          "11": -45,
          "12": 70,
          "13": 70,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 300
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      }
    ]
  },
  "rl": {
    "name": "rl",
    "label": "Roll",
    "description": "Leans to side and rolls smoothly",
    "kind": "behavior",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -70,
          "9": 10,
          "10": -70,
          "11": 10,
          "12": 110,
          "13": 20,
          "14": 110,
          "15": 20
        },
        "speed_deg_per_step": 10,
        "delay_ms": 500
      },
      {
        "angles": {
          "0": 0,
          "8": -55,
          "9": -55,
          "10": 55,
          "11": 55,
          "12": 60,
          "13": 60,
          "14": 60,
          "15": 60
        },
        "speed_deg_per_step": 10,
        "delay_ms": 400
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 300
      }
    ]
  },
  "scrh": {
    "name": "scrh",
    "label": "Scratch",
    "description": "Sits and scratches ear with rear leg",
    "kind": "behavior",
    "frames": [
      {
        "angles": {
          "0": -35,
          "8": -30,
          "9": -30,
          "10": 80,
          "11": -30,
          "12": 40,
          "13": 40,
          "14": 75,
          "15": 45
        },
        "speed_deg_per_step": 10,
        "delay_ms": 200
      },
      {
        "angles": {
          "0": -35,
          "8": -30,
          "9": -30,
          "10": 80,
          "11": 30,
          "12": 40,
          "13": 40,
          "14": 75,
          "15": 95
        },
        "speed_deg_per_step": 20,
        "delay_ms": 150
      },
      {
        "angles": {
          "0": -35,
          "8": -30,
          "9": -30,
          "10": 80,
          "11": -20,
          "12": 40,
          "13": 40,
          "14": 75,
          "15": 55
        },
        "speed_deg_per_step": 20,
        "delay_ms": 150
      },
      {
        "angles": {
          "0": -35,
          "8": -30,
          "9": -30,
          "10": 80,
          "11": 30,
          "12": 40,
          "13": 40,
          "14": 75,
          "15": 95
        },
        "speed_deg_per_step": 20,
        "delay_ms": 150
      },
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": -30,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": 40,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 10,
        "delay_ms": 250
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      }
    ]
  },
  "tbl": {
    "name": "tbl",
    "label": "Table",
    "description": "Level horizontal standing platform",
    "kind": "behavior",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -40,
          "9": -40,
          "10": -40,
          "11": -40,
          "12": 75,
          "13": 75,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 6,
        "delay_ms": 600
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 300
      }
    ]
  },
  "ts": {
    "name": "ts",
    "label": "Test Legs",
    "description": "Tests each leg in round-robin sequence",
    "kind": "behavior",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": 20,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 40,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": 20,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 40,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -20,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 50,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -20,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 50
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      }
    ]
  },
  "wh": {
    "name": "wh",
    "label": "Wave Hand",
    "description": "Sits and waves left front paw",
    "kind": "behavior",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": -30,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": 40,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      },
      {
        "angles": {
          "0": 0,
          "8": 35,
          "9": -30,
          "10": 80,
          "11": 80,
          "12": 0,
          "13": 40,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 12,
        "delay_ms": 200
      },
      {
        "angles": {
          "0": 0,
          "8": 15,
          "9": -30,
          "10": 80,
          "11": 80,
          "12": -30,
          "13": 40,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 12,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": 35,
          "9": -30,
          "10": 80,
          "11": 80,
          "12": 10,
          "13": 40,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 12,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": 15,
          "9": -30,
          "10": 80,
          "11": 80,
          "12": -30,
          "13": 40,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 12,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": -30,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": 40,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      }
    ]
  },
  "bx": {
    "name": "bx",
    "label": "Box",
    "description": "Boxing jabs with front paws",
    "kind": "behavior",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": -30,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": 40,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      },
      {
        "angles": {
          "0": 0,
          "8": 40,
          "9": 10,
          "10": 80,
          "11": 80,
          "12": 10,
          "13": 40,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 16,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": 10,
          "9": 40,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": 10,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 16,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": 40,
          "9": 10,
          "10": 80,
          "11": 80,
          "12": 10,
          "13": 40,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 16,
        "delay_ms": 180
      },
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": -30,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": 40,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      }
    ]
  },
  "cmh": {
    "name": "cmh",
    "label": "Come Here",
    "description": "Beckoning motion with front paw",
    "kind": "behavior",
    "frames": [
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": -30,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": 40,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      },
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": 30,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": -10,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 10,
        "delay_ms": 250
      },
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": 15,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": -35,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 12,
        "delay_ms": 200
      },
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": 30,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": -10,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 12,
        "delay_ms": 200
      },
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": 15,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": -35,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 12,
        "delay_ms": 200
      },
      {
        "angles": {
          "0": 0,
          "8": -30,
          "9": -30,
          "10": 80,
          "11": 80,
          "12": 40,
          "13": 40,
          "14": 75,
          "15": 75
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      },
      {
        "angles": {
          "0": 0,
          "8": -45,
          "9": -45,
          "10": -45,
          "11": -45,
          "12": 80,
          "13": 80,
          "14": 80,
          "15": 80
        },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      }
    ]
  },
  "stair_step_up": {
    "name": "stair_step_up",
    "label": "Stair Step Up (18mm)",
    "description": "4-stage dynamic hoist onto an 18mm step riser",
    "kind": "obstacle",
    "frames": [
      {
        "angles": { "0": 0, "8": -35, "9": -35, "10": -50, "11": -50, "12": 102, "13": 102, "14": 75, "15": 75 },
        "speed_deg_per_step": 10,
        "delay_ms": 280
      },
      {
        "angles": { "0": 0, "8": -55, "9": -55, "10": -45, "11": -45, "12": 85, "13": 85, "14": 75, "15": 75 },
        "speed_deg_per_step": 8,
        "delay_ms": 300
      },
      {
        "angles": { "0": 0, "8": -45, "9": -45, "10": -30, "11": -30, "12": 80, "13": 80, "14": 102, "15": 102 },
        "speed_deg_per_step": 10,
        "delay_ms": 300
      },
      {
        "angles": { "0": 0, "8": -45, "9": -45, "10": -45, "11": -45, "12": 80, "13": 80, "14": 80, "15": 80 },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      }
    ]
  },
  "ramp_climb": {
    "name": "ramp_climb",
    "label": "Ramp Incline Climb",
    "description": "Low-center-of-mass forward gait with pitch compensation for 18° incline ramps",
    "kind": "obstacle",
    "frames": [
      {
        "angles": { "0": 0, "8": -55, "9": -45, "10": -40, "11": -50, "12": 70, "13": 85, "14": 90, "15": 75 },
        "speed_deg_per_step": 10,
        "delay_ms": 220
      },
      {
        "angles": { "0": 0, "8": -40, "9": -55, "10": -50, "11": -40, "12": 85, "13": 70, "14": 75, "15": 90 },
        "speed_deg_per_step": 10,
        "delay_ms": 220
      },
      {
        "angles": { "0": 0, "8": -55, "9": -40, "10": -40, "11": -50, "12": 70, "13": 85, "14": 90, "15": 75 },
        "speed_deg_per_step": 10,
        "delay_ms": 220
      },
      {
        "angles": { "0": 0, "8": -50, "9": -50, "10": -45, "11": -45, "12": 75, "13": 75, "14": 85, "15": 85 },
        "speed_deg_per_step": 8,
        "delay_ms": 200
      }
    ]
  },
  "low_tunnel_crawl": {
    "name": "low_tunnel_crawl",
    "label": "Low Tunnel Belly Crawl",
    "description": "Low-profile belly crawl gait maintaining under 60mm clearance for low-ceiling tunnels",
    "kind": "obstacle",
    "frames": [
      {
        "angles": { "0": 0, "8": -55, "9": -55, "10": 55, "11": 55, "12": 50, "13": 50, "14": 50, "15": 50 },
        "speed_deg_per_step": 8,
        "delay_ms": 300
      },
      {
        "angles": { "0": 10, "8": -45, "9": -65, "10": 45, "11": 65, "12": 55, "13": 45, "14": 55, "15": 45 },
        "speed_deg_per_step": 10,
        "delay_ms": 250
      },
      {
        "angles": { "0": -10, "8": -65, "9": -45, "10": 65, "11": 45, "12": 45, "13": 55, "14": 45, "15": 55 },
        "speed_deg_per_step": 10,
        "delay_ms": 250
      },
      {
        "angles": { "0": 0, "8": -55, "9": -55, "10": 55, "11": 55, "12": 50, "13": 50, "14": 50, "15": 50 },
        "speed_deg_per_step": 8,
        "delay_ms": 250
      }
    ]
  },
  "slalom_weave_left": {
    "name": "slalom_weave_left",
    "label": "Slalom Weave Left",
    "description": "Coordinated head scan left and differential leg stance for precision slalom cone avoidance",
    "kind": "obstacle",
    "frames": [
      {
        "angles": { "0": 30, "8": -35, "9": -55, "10": -40, "11": -50, "12": 85, "13": 75, "14": 80, "15": 80 },
        "speed_deg_per_step": 10,
        "delay_ms": 250
      },
      {
        "angles": { "0": 0, "8": -45, "9": -45, "10": -45, "11": -45, "12": 80, "13": 80, "14": 80, "15": 80 },
        "speed_deg_per_step": 8,
        "delay_ms": 200
      }
    ]
  },
  "slalom_weave_right": {
    "name": "slalom_weave_right",
    "label": "Slalom Weave Right",
    "description": "Coordinated head scan right and differential leg stance for precision slalom cone avoidance",
    "kind": "obstacle",
    "frames": [
      {
        "angles": { "0": -30, "8": -55, "9": -35, "10": -50, "11": -40, "12": 75, "13": 85, "14": 80, "15": 80 },
        "speed_deg_per_step": 10,
        "delay_ms": 250
      },
      {
        "angles": { "0": 0, "8": -45, "9": -45, "10": -45, "11": -45, "12": 80, "13": 80, "14": 80, "15": 80 },
        "speed_deg_per_step": 8,
        "delay_ms": 200
      }
    ]
  }
};
