import gymnasium as gym  # Entorno de simulación CarRacing
from gymnasium.vector import AsyncVectorEnv  # Para evaluación paralela de entornos
import numpy as np  # Operaciones numéricas y manejo de arreglos
import matplotlib.pyplot as plt  # Visualización de resultados
import random  # Funciones aleatorias para selección y mutación
import time  # Medición de tiempo de ejecución
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# crear un video visualizando el mejor individuo de varias pruebas de experimentos.
# Pruebas

# ================================
# Prueba A — Exploración amplia
# Datos / Parámetros:
# pop_size = 80
# generations = 50
# num_steps = 200
# action_repeat = 4
# mutation_rate = 0.12
# elite_size = 4
# selection = torneo (k=3)
# crossover = 1-point
# repetitions = 3–5

# Objetivo / Explicación:
# Evaluar si una mayor diversidad inicial compensa menos iteraciones.
# Gran población introduce diversidad; pocas generaciones limitan refinamiento.
# Mutación alta para mantener exploración.
# Se espera alta varianza entre corridas y tiempo por generación alto.

# ================================
# Prueba B — Profundización
# Datos / Parámetros:
# pop_size = 30
# generations = 100
# num_steps = 200
# action_repeat = 4
# mutation_rate = 0.06
# elite_size = 2
# selection = torneo (k=3)
# crossover = 1-point
# repetitions = 3–5

# Objetivo / Explicación:
# Evaluar si el refinamiento iterativo supera la exploración amplia.
# Poca población pero muchas generaciones permite pulir soluciones.
# Mutación más baja para preservar mejores genes.
# Se espera convergencia más estable; riesgo de estancamiento si la población es homogénea.

# ================================
# Prueba C — Alta resolución de control
# Datos / Parámetros:
# pop_size = 50
# generations = 70
# num_steps = 2000
# action_repeat = 2
# mutation_rate = 0.08
# elite_size = 2
# selection = torneo (k=3)
# crossover = 1-point
# repetitions = 3–5

# Objetivo / Explicación:
# Evaluar si un gran número de decisiones mejora desempeño.
# Mayor resolución de pasos permite maniobras más finas y mejor fitness si el AG explora bien.
# Mayor coste computacional por individuo; contraste con Pruebas A y B.


# ================================
# Función auxiliar para crear entornos de CarRacing
# ================================
def make_env():
    """
    Crea una función que inicializa un entorno CarRacing-v3 con parámetros fijos.
    Útil para la evaluación paralela.
    """
    def _thunk():
        return gym.make("CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=True)
    return _thunk

# ================================
# Representación genética
# ================================
def create_individual(num_steps):
    """
    Crea un individuo (cromosoma) para el algoritmo genético.
    Cada individuo es una secuencia de 'num_steps' acciones aleatorias.
    Cada acción tiene 3 valores: [steer, gas, brake].
    - steer: entre -1 y 1
    - gas: entre 0 y 1
    - brake: entre 0 y 1
    """
    return np.column_stack([
        np.random.uniform(-1, 1, num_steps), # steer
        np.random.uniform(0, 1, num_steps), # gas
        np.random.uniform(0, 1, num_steps), # brake
    ])


def expand_plan(individual, action_repeat):
    """
    Expande el plan de acciones de un individuo repitiendo cada acción 'action_repeat' veces.
    Esto permite que cada decisión macro abarque varios frames.
    """
    # (D,3) -> (D*ACTION_REPEAT, 3)
    return np.repeat(individual, action_repeat, axis=0)

def init_population(pop_size, num_steps):
    """
    Inicializa la población con 'pop_size' individuos, cada uno con 'num_steps' acciones.
    """
    return [create_individual(num_steps) for _ in range(pop_size)]

# ================================
# Evaluation (parallel, early stopping)
# ================================
def evaluate_population_parallel(population, action_repeat, neg_streak_limit=80, max_envs=8):
    """
    Evalúa la población de individuos en paralelo usando varios entornos.
    Permite early stopping si un individuo acumula demasiadas recompensas negativas seguidas.
    Parámetros:
    - population: lista de individuos a evaluar.
    - action_repeat: cuántos frames dura cada acción.
    - neg_streak_limit: límite de pasos negativos antes de terminar el episodio.
    - max_envs: número máximo de entornos paralelos.
    Retorna:
    - Lista de puntajes (fitness) para cada individuo.
    """
    n = len(population)
    if n == 0:
        return []

    # Pre-expande los planes de acción
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
def tournament_selection(population, fitness, k=3):
    """
    Selección por torneo: elige aleatoriamente 'k' individuos y retorna el de mayor fitness.
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
    Cruza dos padres usando 1-point crossover (un solo punto de corte).
    Retorna dos hijos combinando partes de ambos padres.
    """
    n = len(parent1)
    point = np.random.randint(1, n - 1)
    child1 = np.vstack([parent1[:point], parent2[point:]])
    child2 = np.vstack([parent2[:point], parent1[point:]])
    return child1, child2


def mutate(individual, mutation_rate=0.05):
    """
    Aplica mutación sobre un individuo.
    Cada gen (acción) tiene 'mutation_rate' de probabilidad de ser alterado aleatoriamente.
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
    Crea la siguiente generación de la población usando:
    - Elitismo: los mejores 'elite_size' individuos pasan directo.
    - El resto se genera por torneo, crossover y mutación.
    Parámetros:
    - population: lista de individuos.
    - fitness: lista con fitness de cada uno.
    - elite_size: número de mejores que pasan directo.
    - mutation_rate: probabilidad de mutación por gen.
    """
    fitness = np.asarray(fitness)
    n = len(population)

    # Elitismo: selecciona los mejores
    elite_idx = np.argpartition(fitness, -elite_size)[-elite_size:]
    elite_idx = elite_idx[np.argsort(fitness[elite_idx])[::-1]]
    new_pop = [population[i].copy() for i in elite_idx] 

    # Resto de la población
    while len(new_pop) < n:
        # Selección de padres por torneo
        p1 = tournament_selection(population, fitness)
        p2 = tournament_selection(population, fitness)

        # Cruce
        c1, c2 = crossover(p1, p2)
        
        # Mutación
        c1 = mutate(c1, mutation_rate)
        c2 = mutate(c2, mutation_rate)
        new_pop.extend([c1, c2])
    return new_pop[:n]


# ================================
# GA main
# ================================
def genetic_algorithm(
    generations=10,
    pop_size=32,
    num_steps=30,
    action_repeat=4,
    mutation_rate=0.08,
    elite_size=2,
    neg_streak_limit=80,
):
    """
    Ejecuta el Algoritmo Genético sobre CarRacing-v3 usando evaluación paralela y macro-acciones.
    Parámetros:
    - generations: número de generaciones a ejecutar.
    - pop_size: tamaño de la población.
    - num_steps: número de decisiones macro por episodio.
    - action_repeat: frames por decisión.
    - mutation_rate: probabilidad de mutación por gen.
    - elite_size: número de élites.
    - neg_streak_limit: límite de pasos negativos antes de terminar.
    Retorna:
    - best_individual (np.ndarray): mejor secuencia de macro-acciones (num_steps, 3)
    - history (dict): historial de fitness máximo y promedio por generación
    """
    population = init_population(pop_size, num_steps)

    max_fitness_history = []
    avg_fitness_history = []
    gen_times = []

    for gen in range(generations):
        gen_start = time.time() 
        fitness_scores = evaluate_population_parallel(population, action_repeat=action_repeat, neg_streak_limit=neg_streak_limit)

        gen_time = time.time() - gen_start 
        gen_times.append(gen_time)  

        max_fitness = float(np.max(fitness_scores))
        avg_fitness = float(np.mean(fitness_scores))
        max_fitness_history.append(max_fitness)
        avg_fitness_history.append(avg_fitness)

        print(f"Gen {gen+1:>3}/{generations} | Max: {max_fitness:7.2f} | Avg: {avg_fitness:7.2f} | Tiempo: {gen_time:.2f}s")

        population = next_generation(population, fitness_scores, elite_size=elite_size, mutation_rate=mutation_rate)

    # Evaluación final para elegir el mejor individuo
    final_fitness = evaluate_population_parallel(population, action_repeat=action_repeat, neg_streak_limit=neg_streak_limit)
    best_idx = int(np.argmax(final_fitness))
    best_individual = population[best_idx].copy()

    history = {
        "max_fitness": max_fitness_history,
        "avg_fitness": avg_fitness_history,
        "gen_times": gen_times,
    }
    return best_individual, history

# ================================
# Rendering (single env, smooth)
# ================================
def render_individual_realtime_smooth(individual: np.ndarray, action_repeat: int = 4):
    """
    Visualiza en tiempo real la ejecución de un individuo en el entorno CarRacing-v3.
    Muestra el entorno usando matplotlib y actualiza el frame en cada paso.
    Parámetros:
    - individual: secuencia de acciones a ejecutar.
    - action_repeat: cuántos frames dura cada acción.
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
# Funciones para graficar las pruebas experimentales
# ================================

def plot_fitness_curves(all_histories, gens):
    """
    Grafica el progreso del fitness máximo y promedio por generación,
    mostrando la media y desviación estándar entre corridas.
    """
    # Convertir en matrices [n_runs, generations]
    max_fitness_runs = np.array([h["max_fitness"] for h in all_histories])
    avg_fitness_runs = np.array([h["avg_fitness"] for h in all_histories])

    gens_range = np.arange(1, gens+1)

    # Promedio y desviación estándar
    max_mean = np.mean(max_fitness_runs, axis=0)
    max_std  = np.std(max_fitness_runs, axis=0)
    avg_mean = np.mean(avg_fitness_runs, axis=0)
    avg_std  = np.std(avg_fitness_runs, axis=0)

    plt.figure(figsize=(10, 6))
    # Fitness máximo
    plt.plot(gens_range, max_mean, label="Fitness máximo (media)", color="blue")
    plt.fill_between(gens_range, max_mean-max_std, max_mean+max_std, color="blue", alpha=0.2)
    # Fitness promedio
    plt.plot(gens_range, avg_mean, label="Fitness promedio (media)", color="orange")
    plt.fill_between(gens_range, avg_mean-avg_std, avg_mean+avg_std, color="orange", alpha=0.2)

    plt.xlabel("Generación")
    plt.ylabel("Fitness")
    plt.title("Evolución del Fitness (media ± desviación estándar)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()


def plot_execution_times(all_histories):
    """
    Grafica el tiempo de cada corrida para comparar consumo de CPU.
    """
    run_times = [np.sum(h["gen_times"]) for h in all_histories]
    runs = np.arange(1, len(run_times)+1)

    plt.figure(figsize=(8, 5))
    plt.bar(runs, run_times, color="green", alpha=0.7)
    plt.xlabel("Corrida")
    plt.ylabel("Tiempo total (s)")
    plt.title("Tiempo total de cada corrida")
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.show()



# ================================
# Simple console menu
# ================================
def main_menu():
    """
    Menú principal de consola para interactuar con el algoritmo genético.
    Permite:
    1. Ejecutar el algoritmo genético con parámetros configurables.
    2. Visualizar el mejor individuo encontrado.
    3. Graficar la evolución del fitness.
    4. Salir del programa.
    """
    best_individual = None
    history = None
    defaults = {
        "generations": 50,
        "pop_size": 30,
        "num_steps": 30,
        "action_repeat": 4,
        "mutation_rate": 0.08,
        "elite_size": 2,
        "neg_streak_limit": 80,
    }

    while True:
        print("\n--- MENÚ PRINCIPAL ---")
        print("1. Ejecutar Algoritmo Genético")
        print("2. Visualizar mejor individuo")
        print("3. Graficar evolución del fitness")
        print("4. Pruebas experimentales")
        print("5. Salir")
        opt = input("Elige una opción: ").strip()

        if opt == "1":
            try:
                gens = input(f"Generaciones [{defaults['generations']}]: ").strip()
                pop  = input(f"Tamaño de población [{defaults['pop_size']}]: ").strip()
                decs = input(f"Numero de pasos por episodio [{defaults['num_steps']}]: ").strip()
                arep = input(f"Repeticiones por acción [{defaults['action_repeat']}]: ").strip()
                mut  = input(f"Tasa de mutación [{defaults['mutation_rate']}]: ").strip()
                elite = input(f"Tamaño de élite [{defaults['elite_size']}]: ").strip()
                nsl  = input(f"Límite de racha negativa [{defaults['neg_streak_limit']}]: ").strip()

                gens = int(gens) if gens else defaults['generations']
                pop  = int(pop) if pop else defaults['pop_size']
                decs = int(decs) if decs else defaults['num_steps']
                arep = int(arep) if arep else defaults['action_repeat']
                mut  = float(mut) if mut else defaults['mutation_rate']
                elite = int(elite) if elite else defaults['elite_size']
                nsl  = int(nsl) if nsl else defaults['neg_streak_limit']
            except ValueError:
                print("Entrada inválida. Usando valores por defecto.")
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
            print(f"\nAlgoritmo genético finalizado en {dur:.1f}s")

        elif opt == "2":
            if best_individual is None:
                print("Primero ejecuta el algoritmo genético.")
            else:
                arep = input("Frames por decisión para visualización [4]: ").strip()
                arep = int(arep) if arep else 4
                render_individual_realtime_smooth(best_individual, action_repeat=arep)

        elif opt == "3":
            if history is None:
                print("Primero ejecuta el algoritmo genético.")
            else:
                plt.figure()
                plt.plot(history["max_fitness"], label="Fitness máximo")
                plt.plot(history["avg_fitness"], label="Fitness promedio")
                plt.xlabel("Generación")
                plt.ylabel("Fitness")
                plt.title("Evolución del fitness")
                plt.legend()
                plt.show()
        elif opt == "4":
            # ========================
            # Configuración de pruebas experimentales
            # ========================
            try:
                n_runs = int(input("Cantidad de repeticiones de la prueba: ").strip())
            except ValueError:
                n_runs = 3
                print("Valor inválido, usando 3 repeticiones por defecto.")

            # Pedir parámetros del AG
            try:
                gens = input(f"Generaciones [{defaults['generations']}]: ").strip()
                pop  = input(f"Tamaño de población [{defaults['pop_size']}]: ").strip()
                decs = input(f"Numero de pasos por episodio [{defaults['num_steps']}]: ").strip()
                arep = input(f"Repeticiones por acción [{defaults['action_repeat']}]: ").strip()
                mut  = input(f"Tasa de mutación [{defaults['mutation_rate']}]: ").strip()
                elite = input(f"Tamaño de élite [{defaults['elite_size']}]: ").strip()
                nsl  = input(f"Límite de racha negativa [{defaults['neg_streak_limit']}]: ").strip()

                gens = int(gens) if gens else defaults['generations']
                pop  = int(pop) if pop else defaults['pop_size']
                decs = int(decs) if decs else defaults['num_steps']
                arep = int(arep) if arep else defaults['action_repeat']
                mut  = float(mut) if mut else defaults['mutation_rate']
                elite = int(elite) if elite else defaults['elite_size']
                nsl  = int(nsl) if nsl else defaults['neg_streak_limit']
            except ValueError:
                print("Entrada inválida. Usando valores por defecto.")
                gens = defaults['generations']
                pop  = defaults['pop_size']
                decs = defaults['num_steps']
                arep = defaults['action_repeat']
                mut  = defaults['mutation_rate']
                elite = defaults['elite_size']
                nsl  = defaults['neg_streak_limit']

            all_histories = []
            best_overall = None
            best_score_overall = -np.inf

            for run in range(n_runs):
                print(f"\n=== Ejecución {run+1}/{n_runs} ===")
                seed = int(time.time()) + run
                np.random.seed(seed)
                random.seed(seed)

                start = time.time()
                best_ind, hist = genetic_algorithm(
                    generations=gens,
                    pop_size=pop,
                    num_steps=decs,
                    action_repeat=arep,
                    mutation_rate=mut,
                    elite_size=elite,
                    neg_streak_limit=nsl,
                )
                dur = time.time() - start
                print(f"Run {run+1} finalizada en {dur:.1f}s")

                # Guardar historial
                all_histories.append(hist)

                # Actualizar mejor individuo global
                max_run_score = max(hist["max_fitness"])
                if max_run_score > best_score_overall:
                    best_score_overall = max_run_score
                    best_overall = best_ind

            # ========================
            # Graficar resultados
            # ========================
            # Graficar curvas de fitness
            plot_fitness_curves(all_histories, gens)

            #Graficar tiempos de ejecución
            plot_execution_times(all_histories)

            # Guardar el mejor individuo global para visualizar
            best_individual = best_overall
            history = None  # Opcional, para no confundir con una corrida individual
            print("\nMejor individuo global listo para visualización.")
        elif opt == "5":
            print("¡Hasta luego!")
            break
        else:
            print("Opción inválida. Intenta de nuevo.")

if __name__ == "__main__":
    main_menu()
