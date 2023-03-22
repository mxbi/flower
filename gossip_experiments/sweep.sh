set -ex
for GOSSIP_COUNT in 1 2 5 10 
do
    python main.py --gossip_count $GOSSIP_COUNT --gossip_segments 1
    python main.py --gossip_count $GOSSIP_COUNT --gossip_segments 3
    python main.py --gossip_count $GOSSIP_COUNT --gossip_segments 10
    python main.py --gossip_count $GOSSIP_COUNT --pga_frequency 10
    # python main.py --gossip_count $GOSSIP_COUNT --pga_frequency 25
done