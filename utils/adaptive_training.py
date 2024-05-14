import subprocess
import time

# Returns an integer representing the temperature of the GPU in degrees Celsius
def check_gpu_temp():
    result = subprocess.run(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'],
                            capture_output=True, text=True)
    return int(result.stdout.strip())



''' Example of how to pause training when the GPU temperature exceeds a certain threshold
while training:
    # Training code here
    if check_gpu_temp() > 80:  # Assuming 80 degrees Celsius is your threshold
        time.sleep(300)  # Pause for 5 minutes
'''

''' Sample Code to sleep every once in a while to allow the GPU to cool down

import time

for epoch in range(num_epochs):
    for batch in train_dataloader:
        outputs = model(batch)
        loss = criterion(outputs, batch.labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        time.sleep(10)  # Pauses for 10 seconds after each batch

'''




temperature_output = check_gpu_temp()

print(f"GPU temperature: {temperature_output} degrees Celsius")
print(temperature_output)
