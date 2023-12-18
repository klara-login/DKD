`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 2023/11/15 13:48:15
// Design Name: 
// Module Name: fixedAdd16_TB
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module fixedAdd16_TB();

reg [15:0] a,b;
wire [15:0] result;

initial begin
    #0
    a = 16'h34CD;
    b = 16'h84CD;
    #100
    a = 16'h0000;
    b = 16'h34CD;
    #100
    $stop;
end

fixedAdd16 FADD1(
    .a(a),
    .b(b),
    .result(result)
);

endmodule
