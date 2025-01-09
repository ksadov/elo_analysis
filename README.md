Some scripts for analyzing chess Elos and making predictions about future performance.

# Collecting data
If you ran the script `get_fide_html.py` followed by `make_csv.py`, you'd end up with a file `elo.csv` containing a table where each row gives the name of a player that has been ranked in the top 100 global players by [FIDE](https://ratings.fide.com/rankings.phtml). But the FIDE website specifies that "no part of this site may be reproduced, stored in a retrieval system or transmitted in any way or by any means (including photocopying, recording or storing it in any medium by electronic means), without the written permission of FIDE International Chess Federation," so make sure to obtain that written permission first.

# Analysis
`plot_top_players.py` will display a plot of the Elo trajectories of the top N players at a given date.
