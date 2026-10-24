# Use it from other devices

You can run saywho on one computer and use it from your phone, a laptop or any other device on the same network.

## Docker

Nothing to change: the Docker images already accept connections from the network. Open
`http://<server-address>:7860` on the other device.

## Installed with install.bat or install.sh

By default saywho only answers on the computer it runs on. To open it up:

1. **Set a password first.** Anyone on your network will be able to reach it, so open `.env` (see
   [Settings](Settings#where-settings-go)) and set `API_KEY` to a long random value.
2. In the same file, change `HOST=127.0.0.1` to `HOST=0.0.0.0`. Save.
3. Restart saywho.
4. **Windows:** the first time, Windows asks whether Python may use the network. Tick **Private networks** and click
   **Allow access**. (If you clicked Cancel by mistake: Start menu > "Allow an app through Windows Firewall", and
   allow Python on private networks.)
   **Linux** with a firewall: `sudo ufw allow 7860/tcp`.
5. Find the computer's address:
   - Windows: open Command Prompt and run `ipconfig`. Look for **IPv4 Address**, for example `192.168.1.20`.
   - macOS: System Settings > Wi-Fi (or Network) > Details. Look for **IP address**.
   - Linux: `hostname -I`.
6. On the other device, open `http://192.168.1.20:7860` (with your address). The page asks for the API key once
   and remembers it.

## From outside your home or office

Don't open port 7860 to the internet directly. Use a VPN such as WireGuard or Tailscale to reach your network, or
put saywho behind a web server with HTTPS. The second is explained in
[DEPLOYMENT.md](https://github.com/anand34577/saywho/blob/main/docs/DEPLOYMENT.md#behind-a-reverse-proxy-https-public-access).
