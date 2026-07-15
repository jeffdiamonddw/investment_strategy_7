import boto3
import pandas as pd

def get_all_arm_spot_prices(region='us-west-2'):
    ec2 = boto3.client('ec2', region_name=region)
    
    # 1. Get all ARM64 instance types
    print("Fetching all ARM64 instance types...")
    instance_info = ec2.describe_instance_types(
        Filters=[{'Name': 'processor-info.supported-architecture', 'Values': ['arm64']}]
    )
    
    # Extract vCPU counts
    instance_map = {
        it['InstanceType']: it['VCpuInfo']['DefaultVCpus'] 
        for it in instance_info['InstanceTypes']
    }
    
    # 2. Fetch Spot prices for these instances
    # Note: describe_spot_price_history can only handle 200 items per call.
    # We batch them to be safe.
    types_list = list(instance_map.keys())
    spot_data = []
    
    print(f"Querying Spot prices for {len(types_list)} instance types...")
    
    for i in range(0, len(types_list), 200):
        batch = types_list[i:i+200]
        response = ec2.describe_spot_price_history(
            InstanceTypes=batch,
            ProductDescriptions=['Linux/UNIX'],
            StartTime=0 # Latest price
        )
        # Store latest price per type
        latest_prices = {}
        for entry in response['SpotPriceHistory']:
            it = entry['InstanceType']
            price = float(entry['SpotPrice'])
            if it not in latest_prices or price < latest_prices[it]:
                latest_prices[it] = price
        spot_data.append(latest_prices)

    # Merge batch results
    final_prices = {k: v for d in spot_data for k, v in d.items()}
    
    # 3. Create DataFrame
    data = []
    for it, vcpu in instance_map.items():
        if it in final_prices:
            price = final_prices[it]
            data.append({
                'instance_type': it,
                'num_cpus': vcpu,
                'price': price,
                'price_per_cpu': price / vcpu
            })
            
    df = pd.DataFrame(data)
    
    # 4. Display results
    print("\n--- ARM64 Spot Price Analysis ---")
    print(df.sort_values(by='price_per_cpu').to_string(index=False))
    
    avg_price_per_cpu = df['price_per_cpu'].mean()
    std_price_per_cpu = df['price_per_cpu'].std()
    print("-" * 60)
    print(f"Average Price per vCPU-hour (equally weighted): ${avg_price_per_cpu:.4f}")
    print(f"Std Price per vCPU-hour (equally weighted): ${std_price_per_cpu:.4f}")

if __name__ == "__main__":
    get_all_arm_spot_prices()