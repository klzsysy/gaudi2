"""Pair the segmented prefix path with the accepted RMW attention baseline."""
import qwen_d2_product_server as server

# The existing harness toggles the prefix entry separately from attention.
# Both sides of this comparison use Plan A inside attention.
server.baseline_fast = server.product_fast
main = server.main
