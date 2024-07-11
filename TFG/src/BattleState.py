# # This class is for the purpose of storing info regarding the state of a Pokémon battle.
# Qué elementos hay en un estado de un combate Pokémon??

# 1 - Pokémon propios activos
# 2 - Pokémon propios en reserva
# 3 - Pokémon propios presentados pero no usados
# 4,5 y 6 - 1,2, y 3 pero rivales

# De un pokémon se debe almacenar:
# Su especie y forma (si altera sus características de combate)
# Aunque sean formas distintas, el equipo rival no puede tener 1 especie repetida
# Movimientos (de los rivales sólo los conocidos)
# Habilidad en cuanto se descubra
# Valor aproximado de los Puntos de Salud actuales
# Cambios de estado
# Modificaciones en las características

# 7 - Clima, Terreno o Barreras activas, así como un recuento de sus turnos restantes.
from poke_env.environment import Battle

class PokemonState:
    def __init__(self):
        self.species = None
        self.moves = []
        self.hp = None
        self.persistent = None
        self.status = []
        self.stat_mods = {}

    def update(self, penv_pokemon):
        self.species = penv_pokemon.species
        self.moves = penv_pokemon.moves
        self.hp = penv_pokemon.current_hp_fraction
        self.persistent = penv_pokemon.status
        self.status = { **penv_pokemon.effects, "must_recharge": penv_pokemon.must_recharge, "preparing_move": penv_pokemon.preparing_move, "protect_counter": penv_pokemon.protect_counter}
        
        self.stat_mods = penv_pokemon.boosts

    def __str__(self):
        return f"{self.species}\nHP:{self.hp}\n{self.persistent}\n{self.moves}\n{self.stat_mods}\n{self.status}"


class BattleState:
    def __init__(self):
        self.own_active = None
        self.own_reserve = []
        
        self.rival_active = None
        self.rival_reserve = []

        self.own_side_conditions = []
        self.rival_side_conditions = []

    def update(self, penv_battle: Battle):
        self.turn = penv_battle.turn
        self.own_active = PokemonState()
        self.own_active.update(penv_battle.active_pokemon)
        self.own_reserve = []
        for p in penv_battle.available_switches:
            pstate = PokemonState()
            pstate.update(p)
            self.own_reserve.append(pstate)

        self.rival_active = PokemonState()
        self.rival_active.update(penv_battle.opponent_active_pokemon)
        self.rival_reserve = []
        # for p in penv_battle.opponent_available_switches:
        #     pstate = PokemonState()
        #     pstate.update(p)
        #     self.rival_reserve.append(pstate)

        self.own_side_conditions = penv_battle.side_conditions
        self.rival_side_conditions = penv_battle.opponent_side_conditions

    def __str__(self):
        active_str = f"Own Active:\n{self.own_active}\n\nRival Active:\n{self.rival_active}"
        own_reserve_str = ""
        rival_reserve_str = ""
        for p in self.own_reserve:
            own_reserve_str += f"{p}\n"
        for p in self.rival_reserve:
            rival_reserve_str += f"{p}\n"

        return f"Turn:{self.turn-1}\n{active_str}\n\nBarriers:\nOwn:{self.own_side_conditions}\nRival:{self.rival_side_conditions}\n\nOwn Reserve:\n{own_reserve_str}\n\nRival Reserve:\n{rival_reserve_str}"