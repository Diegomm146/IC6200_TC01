import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
import numpy as np
import matplotlib.pyplot as plt
import random
import time

# ================================
# Helpers to construct envs
# ================================
def make_env():
    def _thunk():
        return gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=True)
    return _thunk

# ================================
# Genetic representation (macro-actions)
# ================================
def create_individual(num_steps: int) -> np.ndarray:
    """
    Crea un individuo (cromosoma).
    Cada individuo es una secuencia de NUM_STEPS acciones aleatorias.
    """
    # steer entre -1 y 1, gas y brake entre 0 y 1
    return np.column_stack([
        np.random.uniform(-1, 1, num_steps), # steer
        np.random.uniform(0, 1, num_steps), # gas
        np.random.uniform(0, 1, num_steps), # brake
    ])


def expand_plan(individual: np.ndarray, action_repeat: int):
    # (D,3) -> (D*ACTION_REPEAT, 3)
    return np.repeat(individual, action_repeat, axis=0)

def init_population(pop_size: int, num_steps: int):
    """
    Crea la población inicial con 'pop_size' individuos.
    """
    return [create_individual(num_steps) for _ in range(pop_size)]

# ================================
# Evaluation (parallel, early stopping)
# ================================
def evaluate_population_parallel(population: list[np.ndarray], action_repeat: int, neg_streak_limit: int = 80, max_envs: int = 8):
    """
    Evaluate population in chunks to limit the number of subprocesses.
    """
    n = len(population)
    if n == 0:
        return []

    # Pre-expand
    expanded_all = [expand_plan(ind, action_repeat) for ind in population]
    T = len(expanded_all[0])

    scores = np.zeros(n, dtype=np.float32)

    for start in range(0, n, max_envs):
        end = min(start + max_envs, n)
        batch_idx = np.arange(start, end)
        batch = [expanded_all[i] for i in batch_idx]

        envs = AsyncVectorEnv([make_env() for _ in range(len(batch))])
        obs, infos = envs.reset()
        total = np.zeros(len(batch), dtype=np.float32)
        done = np.zeros(len(batch), dtype=bool)
        neg_streak = np.zeros(len(batch), dtype=np.int32)

        for t in range(T):
            a = np.zeros((len(batch), 3), dtype=np.float32)
            active = ~done
            if active.any():
                a[active] = np.stack([batch[i][t] for i in np.where(active)[0]], axis=0)

            obs, reward, terminated, truncated, infos = envs.step(a)
            reward = reward.astype(np.float32)

            total += reward
            neg_streak[reward < 0] += 1
            neg_streak[reward >= 0] = 0
            done |= (terminated | truncated | (neg_streak >= neg_streak_limit))

            if done.all():
                break

        envs.close()
        scores[batch_idx] = total

    return scores.tolist()

# ================================
# Selection / Crossover / Mutation
# ================================
def tournament_selection(population: list[np.ndarray], fitness: np.ndarray, k: int = 3):
    """
    Selección por torneo.

    Parámetros:
    - population: lista de individuos.
    - fitness: lista con fitness de cada individuo.
    - k: tamaño del torneo (por defecto 3).

    Retorna:
    - individuo ganador del torneo.
    """
    selected = random.sample(range(len(population)), k)
    best = max(selected, key=lambda idx: fitness[idx])
    return population[best]

def crossover(parent1, parent2):
    """
    Cruza dos padres usando 1-point crossover.

    Retorna:
    - hijo1, hijo2
    """
    n = len(parent1)
    point = np.random.randint(1, n - 1)
    return (np.vstack([parent1[:point], parent2[point:]]),
            np.vstack([parent2[:point], parent1[point:]]))


def mutate(individual, mutation_rate=0.05):
    """
    Aplica mutación sobre un individuo.

    Cada gen (acción) tiene 'mutation_rate' de chance de cambiar.
    """
    mutant = individual.copy()
    mask = np.random.rand(len(mutant)) < mutation_rate 
    n = mask.sum()
    if n:
        mutant[mask, 0] = np.random.uniform(-1, 1, n)  # steer
        mutant[mask, 1] = np.random.uniform(0, 1, n)   # gas
        mutant[mask, 2] = np.random.uniform(0, 1, n)   # brake
    return mutant


def next_generation(population, fitness, elite_size=2, mutation_rate=0.05):
    """
    Crea la siguiente generación usando:
    - Elitismo (los mejores pasan directo).
    - Torneo + crossover + mutación para el resto.

    Parámetros:
    - population: lista de individuos.
    - fitness: lista con fitness de cada uno.
    - elite_size: número de mejores que pasan directo.
    - mutation_rate: probabilidad de mutación por gen.
    """
    # Ordenamos por fitness
    fitness = np.asarray(fitness)
    n = len(population)

    # Elites
    elite_idx = np.argpartition(fitness, -elite_size)[-elite_size:]
    elite_idx = elite_idx[np.argsort(fitness[elite_idx])[::-1]]
    new_pop = [population[i].copy() for i in elite_idx] 

    # Fill the rest
    while len(new_pop) < n:
        # Selección por torneo
        p1 = tournament_selection(population, fitness)
        p2 = tournament_selection(population, fitness)
        
        # Crossover
        c1, c2 = crossover(p1, p2)
        c1 = mutate(c1, mutation_rate)
        
        # Mutación
        c2 = mutate(c2, mutation_rate)
        
        new_pop.extend([c1, c2])
    return new_pop[:n]


# ================================
# GA main
# ================================
def genetic_algorithm(
    generations: int = 10,
    pop_size: int = 32,
    num_steps: int = 30,
    action_repeat: int = 4,
    mutation_rate: float = 0.08,
    elite_size: int = 2,
    neg_streak_limit: int = 80,
):
    """
    Run the Genetic Algorithm on CarRacing-v3 with parallel evaluation and macro-actions.

    Returns:
      best_individual (np.ndarray): best macro-action sequence (num_steps, 3)
      history (dict): { 'max_fitness': [...], 'avg_fitness': [...] }
    """
    population = init_population(pop_size, num_steps)

    max_fitness_history = []
    avg_fitness_history = []

    for gen in range(generations):
        fitness_scores = evaluate_population_parallel(
            population,
            action_repeat=action_repeat,
            neg_streak_limit=neg_streak_limit,
        )

        max_fit = float(np.max(fitness_scores))
        avg_fit = float(np.mean(fitness_scores))
        max_fitness_history.append(max_fit)
        avg_fitness_history.append(avg_fit)

        print(f"Gen {gen+1:>3}/{generations} | Max: {max_fit:7.2f} | Avg: {avg_fit:7.2f}")

        population = next_generation(population, fitness_scores, elite_size=elite_size, mutation_rate=mutation_rate)

    # Final evaluation to pick best
    final_fitness = evaluate_population_parallel(
        population,
        action_repeat=action_repeat,
        neg_streak_limit=neg_streak_limit,
    )
    best_idx = int(np.argmax(final_fitness))
    best_individual = population[best_idx].copy()

    history = {
        "max_fitness": max_fitness_history,
        "avg_fitness": avg_fitness_history,
    }
    return best_individual, history

# ================================
# Rendering (single env, smooth)
# ================================
def render_individual_realtime_smooth(individual: np.ndarray, action_repeat: int = 4):
    """
    Visualize a single individual with 'rgb_array' rendering outside training.
    """
    env = gym.make(
        "CarRacing-v3",
        render_mode="rgb_array",
        lap_complete_percent=0.95,
        domain_randomize=False,
        continuous=True,
    )
    obs, info = env.reset()
    total_reward = 0.0

    expanded = expand_plan(individual, action_repeat)
    fig, ax = plt.subplots()
    frame = env.render()
    im = ax.imshow(frame)
    ax.axis("off")
    plt.ion()
    plt.show(block=False)

    for action in expanded:
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        im.set_data(env.render())
        fig.canvas.draw_idle()
        plt.pause(0.001)
        if terminated or truncated:
            break

    plt.ioff()
    plt.close(fig)
    env.close()
    print(f"Total reward: {total_reward:.2f}")

# ================================
# Simple console menu
# ================================
def main_menu():
    best_individual = None
    history = None
    defaults = {
        "generations": 10,
        "pop_size": 32,
        "num_steps": 30,
        "action_repeat": 4,
        "mutation_rate": 0.08,
        "elite_size": 2,
        "neg_streak_limit": 80,
    }

    while True:
        print("\n--- MAIN MENU ---")
        print("1. Run Genetic Algorithm (optimized)")
        print("2. Visualize best individual")
        print("3. Plot fitness evolution")
        print("4. Exit")
        opt = input("Choose an option: ").strip()

        if opt == "1":
            try:
                gens = input(f"Generations [{defaults['generations']}]: ").strip()
                pop  = input(f"Population size [{defaults['pop_size']}]: ").strip()
                decs = input(f"Macro-decisions per episode [{defaults['num_steps']}]: ").strip()
                arep = input(f"Action repeat (frames/decision) [{defaults['action_repeat']}]: ").strip()
                mut  = input(f"Mutation rate [{defaults['mutation_rate']}]: ").strip()
                elite = input(f"Elite size [{defaults['elite_size']}]: ").strip()
                nsl  = input(f"Negative streak limit [{defaults['neg_streak_limit']}]: ").strip()

                gens = int(gens) if gens else defaults['generations']
                pop  = int(pop) if pop else defaults['pop_size']
                decs = int(decs) if decs else defaults['num_steps']
                arep = int(arep) if arep else defaults['action_repeat']
                mut  = float(mut) if mut else defaults['mutation_rate']
                elite = int(elite) if elite else defaults['elite_size']
                nsl  = int(nsl) if nsl else defaults['neg_streak_limit']
            except ValueError:
                print("Invalid input. Using defaults.")
                gens = defaults['generations']
                pop  = defaults['pop_size']
                decs = defaults['num_steps']
                arep = defaults['action_repeat']
                mut  = defaults['mutation_rate']
                elite = defaults['elite_size']
                nsl  = defaults['neg_streak_limit']

            start = time.time()
            best_individual, history = genetic_algorithm(
                generations=gens,
                pop_size=pop,
                num_steps=decs,
                action_repeat=arep,
                mutation_rate=mut,
                elite_size=elite,
                neg_streak_limit=nsl,
            )
            dur = time.time() - start
            print(f"\nGA finished in {dur:.1f}s")

        elif opt == "2":
            if best_individual is None:
                print("Run the GA first.")
            else:
                arep = input("Action repeat for visualization [4]: ").strip()
                arep = int(arep) if arep else 4
                render_individual_realtime_smooth(best_individual, action_repeat=arep)

        elif opt == "3":
            if history is None:
                print("Run the GA first.")
            else:
                plt.figure()
                plt.plot(history["max_fitness"], label="Max fitness")
                plt.plot(history["avg_fitness"], label="Avg fitness")
                plt.xlabel("Generation")
                plt.ylabel("Fitness")
                plt.title("Fitness evolution")
                plt.legend()
                plt.show()

        elif opt == "4":
            print("Bye!")
            break
        else:
            print("Invalid option. Try again.")

if __name__ == "__main__":
    main_menu()
