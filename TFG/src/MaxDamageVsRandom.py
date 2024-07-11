import asyncio
import time

from poke_env.player import Player, RandomPlayer
from poke_env import AccountConfiguration, LocalhostServerConfiguration, ShowdownServerConfiguration, ServerConfiguration
from poke_env.exceptions import ShowdownException

from typing import Optional
from asyncio import Event
from time import perf_counter

import BattleState

ShowdownServerConfiguration = ServerConfiguration(
    "sim.smogon.com:8000", "https://play.pokemonshowdown.com/action.php?"
)
DawnServerConfiguration = ServerConfiguration = ("sim.dawn.com:8000" ,"https://play.pokemonshowdown.com/action.php?server=dawn")

class MaxDamagePlayer(Player):
    # Define the type of a battle state variable
    battle_state: BattleState.BattleState
    battle_state = None
    
    def choose_move(self, battle):
        # Update the current_battle with info from battle
        self.battle_state = BattleState.BattleState()
        self.battle_state.update(battle)
        # If the player can attack, it will
        if battle.available_moves:
            # Finds the best move among available ones
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)

        # If no attack is available, a random switch will be made
        else:
            return self.choose_random_move(battle)


async def main():
    start = time.time()

    # We create the player instance.
    account_config_1 = AccountConfiguration("PokemonBattl3AIUAH", "winbattles")
    account_config_2 = AccountConfiguration("PokemonBattl2AIUAH", "winbattles")

    max_damage_player = MaxDamagePlayer(
        account_configuration=account_config_1,
        server_configuration=LocalhostServerConfiguration,
        #ShowdownServerConfiguration,
        battle_format="gen4randombattle",
    )                
    random_player = RandomPlayer(
        account_configuration=account_config_2,
        server_configuration=LocalhostServerConfiguration,
        #ShowdownServerConfiguration, 
        battle_format="gen4randombattle",
    )
    # Now, let's evaluate our player
    n_challenges = 1
    # Against a bot
    await max_damage_player.battle_against(random_player, n_battles=n_challenges)
    # Against a human
    # await max_damage_player.send_challenges("Markroww", n_challenges=n_challenges)
    print(
        "Max damage player won %d / %d battles [this took %f seconds]"
        % (
            max_damage_player.n_won_battles, n_challenges, time.time() - start
        )
    )

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())

"""@misc{poke_env,
    author       = {Haris Sahovic},
    title        = {Poke-env: pokemon AI in python},
    url          = {https://github.com/hsahovic/poke-env}
}"""