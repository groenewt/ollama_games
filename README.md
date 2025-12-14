# OLLAMA Games
Benchmark LLM strategic reasoning against formal game-theoretic equilibria.

*Do language models reason strategically, or just pattern match?*

This framework pits LLMs against each other in classic game theory scenarios and measures their behavior against mathematically provable optimal strategies.



## Requirements
1. [Docker Website](https://www.docker.com/)
2. Make (**FOR Windows**: Just do it WSL< Docker already requires this>)

## Getting Started
1. Clone the repository
```bash
git clone https://github.com/groenewt/ollama_games.git 
```
2. Navigate to the repository directory
```bash
cd ollama_games
```
3. Run Make up target
```bash
# View the env.example ( or run 'cp .env.example .env') to see general configurations
# This will run flow to build docker images and initial 'up'
make up 
```
4. Open Application and play some games!
```bash
# PORT IS PYAPP_PORT in .env (default of 2718- mirror forward from Marimo default port)
# probably more accesible to just open your browser and put 'localhost:2718' for url-> but if you wanted to open in terminal
brave-browser http://localhost:2718
#AND FOR BURR UI! (Default port of 7241)
brave-browser http://localhost:7241
```


## Why This Matters
### For AI Safety/Alignment

- Cooperation vs defection tendencies vary wildly by model
- Small models have strong behavioral biases baked in
- Strategic reasoning ≠ pattern matching on game descriptions

### For Economics/Game Theory

- Empirical data on how AI agents behave in strategic scenarios
- Deviations from Nash equilibrium are systematic, not random
- Opens questions about AI in mechanism design

### For Accessibility

1. **CPU-only**: No GPU required
2. **Local-first**: No API calls, no data leaves your machine
3. **Reproducible**: Parquet session logs for analysis 



## Resources
1. [Ollama Website](https://ollama.com/)
2. [Ollama Github](https://github.com/ollama/ollama)
3. [Intro to Game Theory](https://plato.stanford.edu/entries/game-theory/)
4. [More About Game Theory](https://ocw.mit.edu/courses/14.126-game-theory-spring-2024/mit14_126_s24_yildiz-lecture-notes.pdf)
5. [Apache Burr Github](https://github.com/apache/burr)
6. [Apache Burr Website](https://burr.apache.org/)
