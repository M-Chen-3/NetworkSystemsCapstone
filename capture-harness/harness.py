import csv
import os
import subprocess
import time

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver import ActionChains

if __name__ == "__main__":
    try:
        os.mkdir("/harness/captures")
    except:
        pass

    print("=" * 50)
    print("Web Traffic Capture Harness")
    print("Connecting to firefox instance...")
    driver = webdriver.Firefox(service=webdriver.FirefoxService(service_args=['--marionette-port', '2828', '--connect-existing']))
    print("Connected!")
    print("=" * 50)

    while True:
        print("+" * 50)
        print("Enter the website to label to start a new capture session (e.g.: \"facebook\", \"github\") or press enter to finish:")
        label = input()

        if label == "":
            print("\tSession finish. Output files have been written to ./captures")
            break

        print("Resetting firefox...")
        driver.execute_script("window.open('about:blank')")
        for old_tab in driver.window_handles[:-1]:
            driver.switch_to.window(old_tab)
            driver.close()
        driver.switch_to.window(driver.window_handles[0])

        print("Starting tshark to capture traffic...")
        p = subprocess.Popen([
            "tshark",
            "-Y",
            "(ip.src != 172.0.0.0/8 or ip.dst != 172.0.0.0/8) and (ip.src != 172.0.0.0/8 or ip.dst != 172.0.0.0/8)",
            "-T",
            "tabs"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        print("Ready to capture! Navigate to http://localhost:5800/ and use the relevant webpage until enough packets have been captured.")
        print("When complete, press enter.")
        input()

        print("Recording captured packets...")

        p.terminate()

        filename = f"capture-{int(time.time())}-{label}.csv"
        with open(f"/harness/captures/{filename}", "w") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(["time", "source", "dest", "protocol", "size", "description", "website"])

            packets = 0
            for line in p.stdout:
                packet_info = line.decode('utf-8').split("\t")
                csv_writer.writerow([packet_info[1].strip(), packet_info[2].strip(), packet_info[4].strip(), packet_info[5].strip(), packet_info[6].strip(), packet_info[7].strip(), label])
                packets += 1

            print("Captured", packets, f"packets. Writing to ./captures/{filename} ...")
