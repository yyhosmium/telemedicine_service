const http = require('http');
const os = require('os');
const socketIO = require('socket.io');
const nodeStatic = require('node-static');

let fileServer = new(nodeStatic.Server)();
let app = http.createServer((req,res)=>{
    fileServer.serve(req,res);
}).listen(8080, ()=>{
    console.log("server is running...");
});

let io = socketIO(app,{
    cors:{
        origin:"*",
        methods:["GET","POST"]
    }
});
io.sockets.on('connection',socket=>{
    console.log(socket.id);
    function log() {
        let array = ['Message from server:'];
        array.push.apply(array,arguments);
        socket.emit('log',array);
    }
    socket.on('message',message=>{
        log('Client said : ' ,message);
        socket.broadcast.emit('message',message);
    });

    socket.on('create or join',room=>{
        let clientsInRoom = io.sockets.adapter.rooms.get(room);
        let numClients = clientsInRoom ? clientsInRoom.size : 0;
        log('Room ' + room + ' now has ' + numClients + ' client(s)');
        
        if(numClients === 0){
            console.log('create room!');
            socket.join(room);
            log('Client ID ' + socket.id + ' created room ' + room);
            socket.emit('created',room,socket.id);
        }
        else if(numClients===1){
            console.log('join room!');
            log('Client Id' + socket.id + 'joined room' + room);
            io.sockets.in(room).emit('join',room);
            socket.join(room);
            socket.emit('joined',room,socket.id);
            io.sockets.in(room).emit('ready');
        }else{
            socket.emit('full',room);
        }
    });


});
