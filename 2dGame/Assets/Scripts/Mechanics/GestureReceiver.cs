using System.Threading;
using NetMQ;
using NetMQ.Sockets;
using UnityEngine;

public class GestureReceiver : MonoBehaviour
{
    private Thread listenerThread;
    private volatile string moveGesture = "NONE";
    private volatile string jumpGesture = "NONE";
    private bool running = true;

    void Start()
    {
        listenerThread = new Thread(Listen);
        listenerThread.Start();
    }

    void Listen()
    {
        AsyncIO.ForceDotNet.Force();
        using (var subSocket = new SubscriberSocket())
        {
            subSocket.Connect("tcp://localhost:5555");
            subSocket.SubscribeToAnyTopic();

            while (running)
            {
                string message = subSocket.ReceiveFrameString();
                Debug.Log("Received Gesture: " + message);
                string[] parts = message.Split('+');
                if (parts.Length == 2)
                {
                    moveGesture = parts[0];
                    jumpGesture = parts[1];
                }
            }
        }
    }

    public string MoveGesture => moveGesture;
    public string JumpGesture => jumpGesture;

    private void OnDestroy()
    {
        running = false;
        listenerThread?.Abort();
    }
}
