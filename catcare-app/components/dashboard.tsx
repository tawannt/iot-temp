// "use client"

// import { useState } from "react"
// import { Button } from "@/components/ui/button"
// import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
// import { Progress } from "@/components/ui/progress"
// import { Cat, Home, BarChart3, User, Settings, Thermometer, Activity, Utensils, Plus, Minus, LogOut } from 'lucide-react'

// import { Line, LineChart, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Legend } from "recharts" // Import Legend
// import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
// // import Chatbot from "@/components/chatbot"

// interface DashboardProps {
//   user: { email: string; name: string } | null
//   onLogout: () => void
// }

// export default function Dashboard({ user, onLogout }: DashboardProps) {
//   const [temperature, setTemperature] = useState(25)
//   const [foodPercentage, setFoodPercentage] = useState(75)
//   const [waterPercentage, setWaterPercentage] = useState(60)
//   const [autoFeeding, setAutoFeeding] = useState(true)
//   const [areaSensor, setAreaSensor] = useState(true)
//   const [laserGame, setLaserGame] = useState(false)
//   const [airConditioner, setAirConditioner] = useState(false) // New state for air conditioner

//   // Generate random data for 24 hours with values from 0 to 100
//   // Temperature will be in Fahrenheit, Humidity in percentage
//   const generateChartData = () => {
//     const data = []
//     for (let i = 0; i <= 24; i++) {
//       data.push({
//         name: i.toString().padStart(2, "0") + ":00",
//         humidity: Math.floor(Math.random() * 101), // Random percentage between 0-100
//         temperature: Math.floor(Math.random() * 101), // Random Fahrenheit between 0-100
//       })
//     }
//     return data
//   }

//   const [chartData] = useState(generateChartData)

//   return (
//     <div className="min-h-screen bg-gray-900 text-white">
//       {/* Sidebar */}
//       <div className="fixed left-0 top-0 h-full w-16 bg-gray-800 border-r border-gray-700 flex flex-col items-center py-4 space-y-6">
//         <div className="p-2">
//           <Cat className="w-8 h-8 text-white" />
//         </div>
//         <nav className="flex flex-col space-y-4">
//           <Button variant="ghost" size="icon" className="text-gray-400 hover:text-white hover:bg-gray-700">
//             <Home className="w-5 h-5" />
//           </Button>
//           <Button variant="ghost" size="icon" className="text-gray-400 hover:text-white hover:bg-gray-700">
//             <BarChart3 className="w-5 h-5" />
//           </Button>
//           <Button variant="ghost" size="icon" className="text-gray-400 hover:text-white hover:bg-gray-700">
//             <User className="w-5 h-5" />
//           </Button>
//           <Button variant="ghost" size="icon" className="text-gray-400 hover:text-white hover:bg-gray-700">
//             <Settings className="w-5 h-5" />
//           </Button>
//         </nav>
//         <div className="mt-auto">
//           <Button
//             variant="ghost"
//             size="icon"
//             onClick={onLogout}
//             className="text-gray-400 hover:text-white hover:bg-gray-700"
//           >
//             <LogOut className="w-5 h-5" />
//           </Button>
//         </div>
//       </div>

//       {/* Main Content */}
//       <div className="ml-16 p-6">
//         {/* Header */}
//         <div className="flex justify-between items-center mb-8">
//           <div>
//             <h1 className="text-2xl font-bold text-white">CatCare Dashboard</h1>
//             <p className="text-gray-400">Cập nhật hệ thống chăm sóc lần cuối: 14:53:34 12/7/2025</p>
//           </div>
//         </div>

//         <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
//           {/* Environmental Monitoring */}
//           <div className="lg:col-span-2">
//             <Card className="bg-gray-800 border-gray-700">
//               <CardHeader>
//                 <CardTitle className="text-white flex items-center">
//                   <Thermometer className="w-5 h-5 mr-2" />
//                   Giám sát môi trường và Thiết bị
//                 </CardTitle>
//               </CardHeader>
//               <CardContent>
//                 <div className="grid grid-cols-3 gap-6">
//                   {/* Temperature */}
//                   <div className="text-center">
//                     <div className="relative w-24 h-24 mx-auto mb-2">
//                       <svg className="w-24 h-24 transform -rotate-90" viewBox="0 0 100 100">
//                         <circle cx="50" cy="50" r="40" stroke="rgb(75, 85, 99)" strokeWidth="8" fill="none" />
//                         <circle
//                           cx="50"
//                           cy="50"
//                           r="40"
//                           stroke="rgb(236, 72, 153)"
//                           strokeWidth="8"
//                           fill="none"
//                           strokeDasharray={`${((temperature - 16) / (30 - 16)) * 251.2} 251.2`}
//                           strokeLinecap="round"
//                         />
//                       </svg>
//                       <div className="absolute inset-0 flex items-center justify-center">
//                         <div className="text-center">
//                           <div className="text-2xl font-bold text-white">{temperature}</div>
//                           <div className="text-xs text-gray-400">°C</div>
//                         </div>
//                       </div>
//                     </div>
//                     <div className="text-sm text-gray-300">Nhiệt độ phòng</div>
//                     <div className="text-xs text-green-400 bg-green-400/20 px-2 py-1 rounded-full mt-1">Tốt</div>
//                   </div>

//                   {/* Humidity */}
//                   <div className="text-center">
//                     <div className="relative w-24 h-24 mx-auto mb-2">
//                       <svg className="w-24 h-24 transform -rotate-90" viewBox="0 0 100 100">
//                         <circle cx="50" cy="50" r="40" stroke="rgb(75, 85, 99)" strokeWidth="8" fill="none" />
//                         <circle
//                           cx="50"
//                           cy="50"
//                           r="40"
//                           stroke="rgb(59, 130, 246)"
//                           strokeWidth="8"
//                           fill="none"
//                           strokeDasharray={`${(45 / 100) * 251.2} 251.2`}
//                           strokeLinecap="round"
//                         />
//                       </svg>
//                       <div className="absolute inset-0 flex items-center justify-center">
//                         <div className="text-center">
//                           <div className="text-2xl font-bold text-white">45</div>
//                           <div className="text-xs text-gray-400">%</div>
//                         </div>
//                       </div>
//                     </div>
//                     <div className="text-sm text-gray-300">Độ ẩm phòng</div>
//                     <div className="text-xs text-green-400 bg-green-400/20 px-2 py-1 rounded-full mt-1">
//                       Bình thường
//                     </div>
//                   </div>

//                   {/* Air Quality */}
//                   <div className="text-center">
//                     <div className="relative w-24 h-24 mx-auto mb-2">
//                       <svg className="w-24 h-24 transform -rotate-90" viewBox="0 0 100 100">
//                         <circle cx="50" cy="50" r="40" stroke="rgb(75, 85, 99)" strokeWidth="8" fill="none" />
//                         <circle
//                           cx="50"
//                           cy="50"
//                           r="40"
//                           stroke="rgb(168, 85, 247)"
//                           strokeWidth="8"
//                           fill="none"
//                           strokeDasharray={`${(1 / 10) * 251.2} 251.2`}
//                           strokeLinecap="round"
//                         />
//                       </svg>
//                       <div className="absolute inset-0 flex items-center justify-center">
//                         <div className="text-center">
//                           <div className="text-2xl font-bold text-white">1</div>
//                           <div className="text-xs text-gray-400">AQI</div>
//                         </div>
//                       </div>
//                     </div>
//                     <div className="text-sm text-gray-300">Chất lượng Không khí</div>
//                     <div className="text-xs text-green-400 bg-green-400/20 px-2 py-1 rounded-full mt-1">Tốt</div>
//                   </div>
//                 </div>

//                 <div className="grid grid-cols-3 gap-6 mt-8">
//                   {/* Activity */}
//                   <div className="text-center">
//                     <div className="relative w-24 h-24 mx-auto mb-2">
//                       <svg className="w-24 h-24 transform -rotate-90" viewBox="0 0 100 100">
//                         <circle cx="50" cy="50" r="40" stroke="rgb(75, 85, 99)" strokeWidth="8" fill="none" />
//                         <circle
//                           cx="50"
//                           cy="50"
//                           r="40"
//                           stroke="rgb(236, 72, 153)"
//                           strokeWidth="8"
//                           fill="none"
//                           strokeDasharray={`${(12 / 20) * 251.2} 251.2`}
//                           strokeLinecap="round"
//                         />
//                       </svg>
//                       <div className="absolute inset-0 flex items-center justify-center">
//                         <div className="text-center">
//                           <div className="text-2xl font-bold text-white">12</div>
//                           <div className="text-xs text-gray-400">lần</div>
//                         </div>
//                       </div>
//                     </div>
//                     <div className="text-sm text-gray-300">Hoạt động trò chơi</div>
//                     <div className="text-xs text-red-400 bg-red-400/20 px-2 py-1 rounded-full mt-1">Không Tốt</div>
//                   </div>

//                   {/* Food Gauge */}
//                   <div className="text-center">
//                     <div className="relative w-24 h-24 mx-auto mb-2">
//                       <svg className="w-24 h-24 transform -rotate-90" viewBox="0 0 100 100">
//                         <circle cx="50" cy="50" r="40" stroke="rgb(75, 85, 99)" strokeWidth="8" fill="none" />
//                         <circle
//                           cx="50"
//                           cy="50"
//                           r="40"
//                           stroke="rgb(59, 130, 246)"
//                           strokeWidth="8"
//                           fill="none"
//                           strokeDasharray={`${(foodPercentage / 100) * 251.2} 251.2`}
//                           strokeLinecap="round"
//                         />
//                       </svg>
//                       <div className="absolute inset-0 flex items-center justify-center">
//                         <div className="text-center">
//                           <div className="text-2xl font-bold text-white">{foodPercentage}</div>
//                           <div className="text-xs text-gray-400">%</div>
//                         </div>
//                       </div>
//                     </div>
//                     <div className="text-sm text-gray-300">Mức thức ăn</div>
//                     <div className="text-xs text-green-400 bg-green-400/20 px-2 py-1 rounded-full mt-1">Đủ</div>
//                   </div>

//                   {/* Water Gauge */}
//                   <div className="text-center">
//                     <div className="relative w-24 h-24 mx-auto mb-2">
//                       <svg className="w-24 h-24 transform -rotate-90" viewBox="0 0 100 100">
//                         <circle cx="50" cy="50" r="40" stroke="rgb(75, 85, 99)" strokeWidth="8" fill="none" />
//                         <circle
//                           cx="50"
//                           cy="50"
//                           r="40"
//                           stroke="rgb(59, 130, 246)"
//                           strokeWidth="8"
//                           fill="none"
//                           strokeDasharray={`${(waterPercentage / 100) * 251.2} 251.2`}
//                           strokeLinecap="round"
//                         />
//                       </svg>
//                       <div className="absolute inset-0 flex items-center justify-center">
//                         <div className="text-center">
//                           <div className="text-2xl font-bold text-white">{waterPercentage}</div>
//                           <div className="text-xs text-gray-400">%</div>
//                         </div>
//                       </div>
//                     </div>
//                     <div className="text-sm text-gray-300">Mức nước</div>
//                     <div className="text-xs text-green-400 bg-green-400/20 px-2 py-1 rounded-full mt-1">Đủ</div>
//                   </div>
//                 </div>
//               </CardContent>
//             </Card>
//           </div>

//           {/* Device Controls */}
//           <div>
//             <Card className="bg-gray-800 border-gray-700">
//               <CardHeader>
//                 <CardTitle className="text-white flex items-center">
//                   <Settings className="w-5 h-5 mr-2" />
//                   Điều khiển thiết bị
//                 </CardTitle>
//               </CardHeader>
//               <CardContent className="space-y-4">
//                 {/* Auto Feeding Toggle */}
//                 <div className="flex items-center justify-between">
//                   <span className="text-gray-300">Cho ăn tự động</span>
//                   <button
//                     onClick={() => setAutoFeeding(!autoFeeding)}
//                     className={`w-12 h-6 rounded-full relative transition-all duration-300 ease-in-out transform hover:scale-105 active:scale-95 ${
//                       autoFeeding ? "bg-blue-600 hover:bg-blue-500" : "bg-gray-600 hover:bg-gray-500"
//                     }`}
//                   >
//                     <div
//                       className={`w-5 h-5 bg-white rounded-full absolute top-0.5 transition-all duration-300 ease-in-out shadow-lg ${
//                         autoFeeding ? "right-0.5 transform" : "left-0.5"
//                       }`}
//                     />
//                   </button>
//                 </div>

//                 {/* Area Sensor Toggle */}
//                 <div className="flex items-center justify-between">
//                   <span className="text-gray-300">Cảm biến giới hạn khu vực</span>
//                   <button
//                     onClick={() => setAreaSensor(!areaSensor)}
//                     className={`w-12 h-6 rounded-full relative transition-all duration-300 ease-in-out transform hover:scale-105 active:scale-95 ${
//                       areaSensor ? "bg-blue-600 hover:bg-blue-500" : "bg-gray-600 hover:bg-gray-500"
//                     }`}
//                   >
//                     <div
//                       className={`w-5 h-5 bg-white rounded-full absolute top-0.5 transition-all duration-300 ease-in-out shadow-lg ${
//                         areaSensor ? "right-0.5" : "left-0.5"
//                       }`}
//                     />
//                   </button>
//                 </div>

//                 {/* Laser Game Toggle */}
//                 <div className="flex items-center justify-between">
//                   <span className="text-gray-300">Trò chơi laser</span>
//                   <button
//                     onClick={() => setLaserGame(!laserGame)}
//                     className={`w-12 h-6 rounded-full relative transition-all duration-300 ease-in-out transform hover:scale-105 active:scale-95 ${
//                       laserGame ? "bg-blue-600 hover:bg-blue-500" : "bg-gray-600 hover:bg-gray-500"
//                     }`}
//                   >
//                     <div
//                       className={`w-5 h-5 bg-white rounded-full absolute top-0.5 transition-all duration-300 ease-in-out shadow-lg ${
//                         laserGame ? "right-0.5" : "left-0.5"
//                       }`}
//                     />
//                   </button>
//                 </div>

//                 {/* Air Conditioner Toggle */}
//                 <div className="flex items-center justify-between">
//                   <span className="text-gray-300">Bật/Tắt máy lạnh</span>
//                   <button
//                     onClick={() => setAirConditioner(!airConditioner)}
//                     className={`w-12 h-6 rounded-full relative transition-all duration-300 ease-in-out transform hover:scale-105 active:scale-95 ${
//                       airConditioner ? "bg-blue-600 hover:bg-blue-500" : "bg-gray-600 hover:bg-gray-500"
//                     }`}
//                   >
//                     <div
//                       className={`w-5 h-5 bg-white rounded-full absolute top-0.5 transition-all duration-300 ease-in-out shadow-lg ${
//                         airConditioner ? "right-0.5" : "left-0.5"
//                       }`}
//                     />
//                   </button>
//                 </div>

//                 <div className="mt-6">
//                   <div className="text-gray-300 mb-2">Điều khiển máy lạnh</div>
//                   <button
//                     onClick={() => setTemperature((prev) => Math.min(30, prev + 1))}
//                     className="w-full bg-pink-500 hover:bg-pink-400 active:bg-pink-600 transition-all duration-200 transform hover:scale-[1.02] active:scale-[0.98] rounded-lg p-4 mb-2"
//                   >
//                     <div className="flex items-center justify-center gap-2">
//                       <Plus className="w-4 h-4 text-white" />
//                       <span className="text-white font-bold">TĂNG NHIỆT ĐỘ</span>
//                     </div>
//                   </button>
//                   <button
//                     onClick={() => setTemperature((prev) => Math.max(16, prev - 1))}
//                     className="w-full bg-pink-500 hover:bg-pink-400 active:bg-pink-600 transition-all duration-200 transform hover:scale-[1.02] active:scale-[0.98] rounded-lg p-4"
//                   >
//                     <div className="flex items-center justify-center gap-2">
//                       <Minus className="w-4 h-4 text-white" />
//                       <span className="text-white font-bold">GIẢM NHIỆT ĐỘ</span>
//                     </div>
//                   </button>
//                 </div>

//                 <div className="mt-6 space-y-2 text-sm text-gray-400">
//                   <div>
//                     Trạng thái kết nối: <span className="text-green-400">Đã kết nối</span>
//                   </div>
//                   <div>
//                     Thiết bị hoạt động: <span className="text-white">8/10</span>
//                   </div>
//                   <div>
//                     Mèo trong khu vực: <span className="text-green-400">An toàn</span>
//                   </div>
//                 </div>
//               </CardContent>
//             </Card>
//           </div>
//         </div>

//         {/* Bottom Row */}
//         <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
//           {/* Food and Water Status */}
//           <Card className="bg-gray-800 border-gray-700">
//             <CardHeader>
//               <CardTitle className="text-white flex items-center">
//                 <Utensils className="w-5 h-5 mr-2" />
//                 Trạng thái thức ăn và nước trong bình
//               </CardTitle>
//             </CardHeader>
//             <CardContent className="space-y-6">
//               <div>
//                 <div className="flex justify-between items-center mb-2">
//                   <span className="text-gray-300">Thức ăn khô</span>
//                   <span className="text-green-400">{foodPercentage}%</span>
//                 </div>
//                 <Progress value={foodPercentage} className="h-2 bg-gray-700" />
//                 <div className="flex justify-between text-xs text-gray-400 mt-1">
//                   <span>Lần cuối: 2 giờ trước</span>
//                   <span>Lần tiếp: 4 giờ nữa</span>
//                 </div>
//                 <Button
//                   className="w-full mt-3 bg-purple-600 hover:bg-purple-500 active:bg-purple-700 transition-all duration-200 transform hover:scale-[1.02] active:scale-[0.98] shadow-lg hover:shadow-xl"
//                   onClick={() => setFoodPercentage((prev) => Math.max(0, prev - 5))}
//                 >
//                   ▶ Cho thức ăn khô ngay
//                 </Button>
//               </div>

//               <div>
//                 <div className="flex justify-between items-center mb-2">
//                   <span className="text-gray-300">Nước uống</span>
//                   <span className="text-green-400">{waterPercentage}%</span>
//                 </div>
//                 <Progress value={waterPercentage} className="h-2 bg-gray-700" />
//                 <div className="flex justify-between text-xs text-gray-400 mt-1">
//                   <span>Lần cuối: 1 giờ trước</span>
//                   <span>Lần tiếp: luôn sẵn sàng</span>
//                 </div>
//                 <Button
//                   className="w-full mt-3 bg-blue-600 hover:bg-blue-500 active:bg-blue-700 transition-all duration-200 transform hover:scale-[1.02] active:scale-[0.98] shadow-lg hover:shadow-xl"
//                   onClick={() => setWaterPercentage((prev) => Math.max(0, prev - 5))}
//                 >
//                   ▶ Cho nước uống ngay
//                 </Button>
//               </div>

//               <div className="text-xs text-gray-400 space-y-1">
//                 <div>⏰ Lịch cho ăn: 7:00, 12:00, 18:00</div>
//                 <div>Số lần cho mèo chơi hôm nay: 7</div>
//                 <div>
//                   Cảnh báo sắp hết: <span className="text-orange-400">Dưới 10%</span>
//                 </div>
//               </div>
//             </CardContent>
//           </Card>

//           {/* Activity Chart */}
//           <Card className="bg-gray-800 border-gray-700">
//             <CardHeader>
//               <CardTitle className="text-white flex items-center">
//                 <Activity className="w-5 h-5 mr-2" />
//                 Biểu đồ hoạt động
//               </CardTitle>
//             </CardHeader>
//             <CardContent>
//               <div className="mb-4">
//                 <h4 className="text-gray-300 mb-2">Biểu đồ thời gian</h4>
//                 <ChartContainer
//                   config={{
//                     humidity: {
//                       label: "Độ ẩm",
//                       color: "cyan", // Match hardcoded color
//                     },
//                     temperature: {
//                       label: "Nhiệt độ",
//                       color: "magenta", // Match hardcoded color
//                     },
//                   }}
//                   className="h-[200px] w-full" // Increased height for better label visibility
//                 >
//                   <ResponsiveContainer width="100%" height="100%">
//                     <LineChart data={chartData} margin={{ top: 5, right: 10, left: 10, bottom: 0 }}>
//                       <CartesianGrid vertical={false} strokeDasharray="3 3" stroke="hsl(var(--muted-foreground))" />
//                       <XAxis
//                         dataKey="name"
//                         tickLine={false}
//                         axisLine={false}
//                         tickMargin={8}
//                         // Format to show only even hours (00, 02, 04, ...)
//                         tickFormatter={(value) => {
//                           const hour = Number.parseInt(value.split(":")[0])
//                           return hour % 2 === 0 ? value.split(":")[0] : ""
//                         }}
//                         className="text-xs text-gray-400"
//                         interval={0} // Force all ticks to be displayed, then formatter filters
//                       />
//                       <YAxis
//                         tickLine={false}
//                         axisLine={false}
//                         tickMargin={8}
//                         className="text-xs text-gray-400"
//                         domain={[0, 100]} // Set Y-axis domain from 0 to 100
//                         ticks={[0, 20, 40, 60, 80, 100]} // Set specific Y-axis ticks
//                       />
//                       <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
//                       <Line
//                         type="monotone"
//                         dataKey="humidity"
//                         stroke="cyan" // Hardcoded bright color for visibility
//                         strokeWidth={4} // Increased stroke width for better visibility
//                         dot={true} // Added dots to ensure data points are visible
//                         name="Độ ẩm" // Name for the legend
//                       />
//                       <Line
//                         type="monotone"
//                         dataKey="temperature"
//                         stroke="magenta" // Hardcoded bright color for visibility
//                         strokeWidth={4} // Increased stroke width for better visibility
//                         dot={true} // Added dots to ensure data points are visible
//                         name="Nhiệt độ" // Name for the legend
//                       />
//                       <Legend // Add Legend component
//                         wrapperStyle={{ paddingTop: 10, fontSize: 12, color: "white" }} // Style for legend wrapper
//                         formatter={(value, entry) => <span style={{ color: entry.color }}>{value}</span>}
//                       />
//                     </LineChart>
//                   </ResponsiveContainer>
//                 </ChartContainer>
//                 {/* Removed old manual legend as Recharts Legend is now used */}
//                 {/* <div className="flex justify-between text-xs text-gray-400 mt-2">
//             <span className="flex items-center gap-1">
//               <span className="w-2 h-2 rounded-full bg-[hsl(var(--chart-1))]"></span> Độ ẩm
//             </span>
//             <span className="flex items-center gap-1">
//               <span className="w-2 h-2 rounded-full bg-[hsl(var(--chart-2))]"></span> Nhiệt độ
//             </span>
//           </div> */}
//               </div>

//               <div>
//                 <h4 className="text-gray-300 mb-2">Biểu đồ theo đối cảnh báo</h4>
//                 <div className="relative w-24 h-24 mx-auto">
//                   <svg className="w-24 h-24 transform -rotate-90" viewBox="0 0 100 100">
//                     <circle cx="50" cy="50" r="40" stroke="rgb(75, 85, 99)" strokeWidth="8" fill="none" />
//                     <circle
//                       cx="50"
//                       cy="50"
//                       r="40"
//                       stroke="rgb(34, 197, 94)"
//                       strokeWidth="8"
//                       fill="none"
//                       strokeDasharray={`${(70 / 100) * 251.2} 251.2`}
//                       strokeLinecap="round"
//                     />
//                     <circle
//                       cx="50"
//                       cy="50"
//                       r="30"
//                       stroke="rgb(168, 85, 247)"
//                       strokeWidth="6"
//                       fill="none"
//                       strokeDasharray={`${(20 / 100) * 188.4} 188.4`}
//                       strokeLinecap="round"
//                     />
//                     <circle
//                       cx="50"
//                       cy="50"
//                       r="20"
//                       stroke="rgb(236, 72, 153)"
//                       strokeWidth="4"
//                       fill="none"
//                       strokeDasharray={`${(10 / 100) * 125.6} 125.6`}
//                       strokeLinecap="round"
//                     />
//                   </svg>
//                 </div>
//                 <div className="text-xs text-gray-400 space-y-1 mt-2">
//                   <div>● Số lần nhận nút chơi</div>
//                   <div>● Số lần ăn</div>
//                   <div>● Số lần uống nước</div>
//                 </div>
//               </div>
//             </CardContent>
//           </Card>
//         </div>
//         {/* <Chatbot /> */}
//       </div>
//     </div>
//   )
// }

"use client"

import { useState, useEffect, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import {
  Cat,
  Home,
  BarChart3,
  User,
  Settings,
  Thermometer,
  Activity,
  Utensils,
  Plus,
  Minus,
  LogOut,
  Wifi,
  WifiOff,
} from "lucide-react"

import { Line, LineChart, XAxis, YAxis, CartesianGrid, ResponsiveContainer } from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { iotDataService, IoTSensorData, IoTHistoryEntry } from "@/lib/iot-service"
import { mqttService } from "@/lib/mqtt-service"

interface DashboardProps {
  user: { email: string; name: string } | null
  onLogout: () => void
}

// Utility functions for air quality conversion (based on ESP32 MQ2 sensor)
const getAirQualityStatus = (voltage: number) => {
  if (voltage < 0.4) return { status: "Không khí sạch", level: 1, color: "green" };
  else if (voltage < 1.2) return { status: "Bình thường", level: 2, color: "green" };
  else if (voltage < 2.0) return { status: "Khí nhẹ", level: 3, color: "yellow" };
  else if (voltage < 2.8) return { status: "Rò rỉ nhẹ!", level: 4, color: "orange" };
  else return { status: "Cảnh báo khí mạnh!", level: 5, color: "red" };
}

export default function Dashboard({ user, onLogout }: DashboardProps) {
  // IoT Sensor Data (from ESP32)
  const [sensorData, setSensorData] = useState<IoTSensorData>({
    food_percentage: 0, // Will be updated from ESP32
    water_percentage: 0, // Will be updated from ESP32
    temperature: 0, // Will be updated from ESP32
    humidity: 0, // Will be updated from ESP32
    air_quality_voltage: 0, // Will be updated from ESP32
    last_updated: Date.now()
  })

  // Alert system status
  const [alertSystemActive, setAlertSystemActive] = useState(true)
  const [lastAlertTime, setLastAlertTime] = useState<number | null>(null)

  // Control states
  const [autoFeeding, setAutoFeeding] = useState(true)
  const [areaSensor, setAreaSensor] = useState(true)
  const [laserGame, setLaserGame] = useState(false)
  const [lastFeedTime, setLastFeedTime] = useState(Date.now())
  
  // UI states
  const [isLoading, setIsLoading] = useState(false)
  const [cloudConnected, setCloudConnected] = useState(false)
  const [historicalData, setHistoricalData] = useState<IoTHistoryEntry[]>([])

  // Destructure sensor data for easier access  
  const { 
    food_percentage: foodPercentage = 0, 
    water_percentage: waterPercentage = 0, 
    temperature = 0, 
    humidity = 0, 
    air_quality_voltage: airQualityVoltage = 0 
  } = sensorData
  
  // Convert voltage to air quality status (with safety check)
  const airQualityInfo = getAirQualityStatus(airQualityVoltage || 0)

  // Initialize IoT Data Service and request notification permission
  useEffect(() => {
    const initializeIoT = async () => {
      try {
        // Initialize IoT service
        await iotDataService.initializeRealDataMode()
        
        // Request notification permission
        await iotDataService.requestNotificationPermission()
        
        // Subscribe to real-time sensor data from ESP32
        const unsubscribeSensor = iotDataService.subscribeSensorData((data: IoTSensorData) => {
          setSensorData(data)
          setCloudConnected(true)
          console.log('📊 Dashboard updated with ESP32 data:', {
            food: `${data.food_percentage || 0}%`,
            water: `${data.water_percentage || 0}%`,
            temp: `${(data.temperature || 0).toFixed(1)}°C`,
            humidity: `${(data.humidity || 0).toFixed(1)}%`,
            air: `${(data.air_quality_voltage || 0).toFixed(2)}V`
          })
        })

        // Load initial historical data
        const history = await iotDataService.getHistoricalData()
        console.log('📊 Initial historical data loaded:', history.length, 'entries')
        setHistoricalData(history)

        // Subscribe to real-time historical data updates
        const unsubscribeHistory = iotDataService.subscribeHistoricalData((data: IoTHistoryEntry[]) => {
          console.log('📊 Real-time historical data updated:', data.length, 'entries')
          setHistoricalData(data)
        })

        // Check ESP32 connection status
        setCloudConnected(iotDataService.isESP32Online())

        // Cleanup function
        return () => {
          unsubscribeSensor()
          unsubscribeHistory()
        }
      } catch (error) {
        console.warn('IoT initialization failed, using local mode:', error)
        setCloudConnected(false)
      }
    }

    initializeIoT()
  }, [])

  // Auto-feeding logic - sends commands to ESP32
  useEffect(() => {
    if (autoFeeding && foodPercentage < 20) {
      const feedingInterval = setInterval(async () => {
        setIsLoading(true)
        try {
          // Send feed command to ESP32
          await mqttService.sendCommand('feed', { amount: 30 })
          setLastFeedTime(Date.now())
          
          // Simulate feeding delay
          setTimeout(() => {
            setIsLoading(false)
          }, 2000)
        } catch (error) {
          console.warn('Auto-feeding command failed:', error)
          setIsLoading(false)
        }
      }, 10000) // Check every 10 seconds

      return () => clearInterval(feedingInterval)
    }
  }, [autoFeeding, foodPercentage])

  // Enhanced control functions - now communicate with ESP32 via cloud
  const handleTemperatureChange = useCallback(async (delta: number) => {
    setIsLoading(true)
    try {
      const newTemp = Math.max(18, Math.min(32, temperature + delta))
      await mqttService.sendCommand('increase_temp', newTemp)
      
      // Simulate command processing time
      setTimeout(() => {
        setIsLoading(false)
      }, 1000)
    } catch (error) {
      console.warn('Temperature control failed:', error)
      setIsLoading(false)
    }
  }, [temperature])

  const handleManualFeed = useCallback(async () => {
    if (isLoading) return
    setIsLoading(true)
    try {
      await mqttService.sendCommand('feed', { amount: 25 })
      setLastFeedTime(Date.now())
      
      // Simulate feeding process
      setTimeout(() => {
        setIsLoading(false)
      }, 2000)
    } catch (error) {
      console.warn('Manual feeding failed:', error)
      setIsLoading(false)
    }
  }, [isLoading])

  const handleRefillWater = useCallback(async () => {
    if (isLoading) return
    setIsLoading(true)
    try {
      await mqttService.sendCommand('refill_water', { amount: 100 })
      
      // Simulate refill process
      setTimeout(() => {
        setIsLoading(false)
      }, 1500)
    } catch (error) {
      console.warn('Water refill failed:', error)
      setIsLoading(false)
    }
  }, [isLoading])

  // Control toggle handlers
  const handleAutoFeedingToggle = useCallback(async () => {
    const newValue = !autoFeeding
    setAutoFeeding(newValue)
    // Note: Auto feeding is handled locally, not sent to ESP32
  }, [autoFeeding])

  const handleAreaSensorToggle = useCallback(async () => {
    const newValue = !areaSensor
    setAreaSensor(newValue)
    // Note: Area sensor is handled locally
  }, [areaSensor])

  const handleLaserGameToggle = useCallback(async () => {
    const newValue = !laserGame
    setLaserGame(newValue)
    // Note: Laser game is handled locally
  }, [laserGame])

  const formatLastFeedTime = (timestamp: number) => {
    const now = Date.now()
    const diff = now - timestamp
    const minutes = Math.floor(diff / 60000)
    const hours = Math.floor(minutes / 60)
    
    if (hours > 0) {
      return `${hours} giờ trước`
    } else if (minutes > 0) {
      return `${minutes} phút trước`
    } else {
      return 'Vừa xong'
    }
  }

  // Auto-scroll bottom toggle
  const [autoScroll, setAutoScroll] = useState(true)

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Sidebar */}
      <div className="fixed left-0 top-0 h-full w-16 bg-gray-800 border-r border-gray-700 flex flex-col items-center py-4 space-y-6">
        <div className="p-2">
          <Cat className="w-8 h-8 text-white" />
        </div>
        <nav className="flex flex-col space-y-4">
          <Button variant="ghost" size="icon" className="text-gray-400 hover:text-white hover:bg-gray-700">
            <Home className="w-5 h-5" />
          </Button>
          <Button variant="ghost" size="icon" className="text-gray-400 hover:text-white hover:bg-gray-700">
            <BarChart3 className="w-5 h-5" />
          </Button>
          <Button variant="ghost" size="icon" className="text-gray-400 hover:text-white hover:bg-gray-700">
            <User className="w-5 h-5" />
          </Button>
          <Button variant="ghost" size="icon" className="text-gray-400 hover:text-white hover:bg-gray-700">
            <Settings className="w-5 h-5" />
          </Button>
        </nav>
        <div className="mt-auto">
          <Button
            variant="ghost"
            size="icon"
            onClick={onLogout}
            className="text-gray-400 hover:text-white hover:bg-gray-700"
          >
            <LogOut className="w-5 h-5" />
          </Button>
        </div>
      </div>

      {/* Main Content */}
      <div className="ml-16 p-6">
        {/* Header */}
        <div className="flex justify-between items-center mb-8">
          <div>
            <h1 className="text-2xl font-bold text-white flex items-center gap-3">
              CatCare Dashboard
              <div className="flex items-center gap-2">
                {cloudConnected ? (
                  <>
                    <Wifi className="w-5 h-5 text-green-400" />
                    <span className="text-sm text-green-400 bg-green-400/20 px-2 py-1 rounded-full">
                      Cloud Connected
                    </span>
                  </>
                ) : (
                  <>
                    <WifiOff className="w-5 h-5 text-orange-400" />
                    <span className="text-sm text-orange-400 bg-orange-400/20 px-2 py-1 rounded-full">
                      ESP32 Offline
                    </span>
                  </>
                )}
              </div>
            </h1>
            <p className="text-gray-400">
              Cập nhật hệ thống chăm sóc lần cuối: {new Date(sensorData.last_updated).toLocaleString('vi-VN')}
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Environmental Monitoring */}
          <div className="lg:col-span-2">
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="text-white flex items-center">
                  <Thermometer className="w-5 h-5 mr-2" />
                  Giám sát môi trường và Thiết bị
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-3 gap-6">
                  {/* Temperature */}
                  <div className="text-center">
                    <div className="relative w-24 h-24 mx-auto mb-2">
                      <svg className="w-24 h-24 transform -rotate-90" viewBox="0 0 100 100">
                        <circle cx="50" cy="50" r="40" stroke="rgb(75, 85, 99)" strokeWidth="8" fill="none" />
                        <circle
                          cx="50"
                          cy="50"
                          r="40"
                          stroke="rgb(236, 72, 153)"
                          strokeWidth="8"
                          fill="none"
                          strokeDasharray={`${(temperature / 50) * 251.2} 251.2`}
                          strokeLinecap="round"
                        />
                      </svg>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-white">{Math.round(temperature)}</div>
                          <div className="text-xs text-gray-400">°C</div>
                        </div>
                      </div>
                    </div>
                    <div className="text-sm text-gray-300">Nhiệt độ phòng</div>
                    <div className="text-xs text-green-400 bg-green-400/20 px-2 py-1 rounded-full mt-1">Tốt</div>
                  </div>

                  {/* Humidity */}
                  <div className="text-center">
                    <div className="relative w-24 h-24 mx-auto mb-2">
                      <svg className="w-24 h-24 transform -rotate-90" viewBox="0 0 100 100">
                        <circle cx="50" cy="50" r="40" stroke="rgb(75, 85, 99)" strokeWidth="8" fill="none" />
                        <circle
                          cx="50"
                          cy="50"
                          r="40"
                          stroke="rgb(59, 130, 246)"
                          strokeWidth="8"
                          fill="none"
                          strokeDasharray={`${(humidity / 100) * 251.2} 251.2`}
                          strokeLinecap="round"
                        />
                      </svg>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-white">{Math.round(humidity)}</div>
                          <div className="text-xs text-gray-400">%</div>
                        </div>
                      </div>
                    </div>
                    <div className="text-sm text-gray-300">Độ ẩm phòng</div>
                    <div className="text-xs text-green-400 bg-green-400/20 px-2 py-1 rounded-full mt-1">
                      {humidity < 30 ? 'Khô' : humidity > 70 ? 'Ẩm' : 'Bình thường'}
                    </div>
                  </div>

                  {/* Air Quality */}
                  <div className="text-center">
                    <div className="relative w-24 h-24 mx-auto mb-2">
                      <svg className="w-24 h-24 transform -rotate-90" viewBox="0 0 100 100">
                        <circle cx="50" cy="50" r="40" stroke="rgb(75, 85, 99)" strokeWidth="8" fill="none" />
                        <circle
                          cx="50"
                          cy="50"
                          r="40"
                          stroke={
                            airQualityInfo.color === 'green' ? 'rgb(34, 197, 94)' :
                            airQualityInfo.color === 'yellow' ? 'rgb(234, 179, 8)' :
                            airQualityInfo.color === 'orange' ? 'rgb(249, 115, 22)' :
                            'rgb(239, 68, 68)'
                          }
                          strokeWidth="8"
                          fill="none"
                          strokeDasharray={`${(airQualityInfo.level / 5) * 251.2} 251.2`}
                          strokeLinecap="round"
                        />
                      </svg>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-white">{(airQualityVoltage || 0).toFixed(2)}</div>
                          <div className="text-xs text-gray-400">Volt</div>
                        </div>
                      </div>
                    </div>
                    <div className="text-sm text-gray-300">Chất lượng Không khí</div>
                    <div className={`text-xs px-2 py-1 rounded-full mt-1 ${
                      airQualityInfo.color === 'green' ? 'text-green-400 bg-green-400/20' :
                      airQualityInfo.color === 'yellow' ? 'text-yellow-400 bg-yellow-400/20' :
                      airQualityInfo.color === 'orange' ? 'text-orange-400 bg-orange-400/20' :
                      'text-red-400 bg-red-400/20'
                    }`}>
                      {airQualityInfo.status}
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-3 gap-6 mt-8">
                  {/* Activity */}
                  <div className="text-center">
                    <div className="relative w-24 h-24 mx-auto mb-2">
                      <svg className="w-24 h-24 transform -rotate-90" viewBox="0 0 100 100">
                        <circle cx="50" cy="50" r="40" stroke="rgb(75, 85, 99)" strokeWidth="8" fill="none" />
                        <circle
                          cx="50"
                          cy="50"
                          r="40"
                          stroke="rgb(236, 72, 153)"
                          strokeWidth="8"
                          fill="none"
                          strokeDasharray={`${(12 / 20) * 251.2} 251.2`}
                          strokeLinecap="round"
                        />
                      </svg>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-white">12</div>
                          <div className="text-xs text-gray-400">lần</div>
                        </div>
                      </div>
                    </div>
                    <div className="text-sm text-gray-300">Hoạt động trò chơi</div>
                    <div className="text-xs text-red-400 bg-red-400/20 px-2 py-1 rounded-full mt-1">Không Tốt</div>
                  </div>

                  {/* Food Gauge */}
                  <div className="text-center">
                    <div className="relative w-24 h-24 mx-auto mb-2">
                      <svg className="w-24 h-24 transform -rotate-90" viewBox="0 0 100 100">
                        <circle cx="50" cy="50" r="40" stroke="rgb(75, 85, 99)" strokeWidth="8" fill="none" />
                        <circle
                          cx="50"
                          cy="50"
                          r="40"
                          stroke="rgb(59, 130, 246)"
                          strokeWidth="8"
                          fill="none"
                          strokeDasharray={`${(foodPercentage / 100) * 251.2} 251.2`}
                          strokeLinecap="round"
                        />
                      </svg>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-white">{Math.round(foodPercentage)}</div>
                          <div className="text-xs text-gray-400">%</div>
                        </div>
                      </div>
                    </div>
                    <div className="text-sm text-gray-300">Thức ăn</div>
                    <div className={`text-xs px-2 py-1 rounded-full mt-1 ${
                      foodPercentage < 20 ? 'text-red-400 bg-red-400/20' : 
                      foodPercentage < 50 ? 'text-yellow-400 bg-yellow-400/20' : 'text-green-400 bg-green-400/20'
                    }`}>
                      {foodPercentage < 20 ? 'Sắp hết' : foodPercentage < 50 ? 'Cần thêm' : 'Đủ'}
                    </div>
                  </div>

                  {/* Water Gauge */}
                  <div className="text-center">
                    <div className="relative w-24 h-24 mx-auto mb-2">
                      <svg className="w-24 h-24 transform -rotate-90" viewBox="0 0 100 100">
                        <circle cx="50" cy="50" r="40" stroke="rgb(75, 85, 99)" strokeWidth="8" fill="none" />
                        <circle
                          cx="50"
                          cy="50"
                          r="40"
                          stroke="rgb(6, 182, 212)"
                          strokeWidth="8"
                          fill="none"
                          strokeDasharray={`${(waterPercentage / 100) * 251.2} 251.2`}
                          strokeLinecap="round"
                        />
                      </svg>
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="text-center">
                          <div className="text-2xl font-bold text-white">{Math.round(waterPercentage)}</div>
                          <div className="text-xs text-gray-400">%</div>
                        </div>
                      </div>
                    </div>
                    <div className="text-sm text-gray-300">Nước</div>
                    <div className={`text-xs px-2 py-1 rounded-full mt-1 ${
                      waterPercentage < 20 ? 'text-red-400 bg-red-400/20' : 
                      waterPercentage < 50 ? 'text-yellow-400 bg-yellow-400/20' : 'text-green-400 bg-green-400/20'
                    }`}>
                      {waterPercentage < 20 ? 'Sắp hết' : waterPercentage < 50 ? 'Cần thêm' : 'Đủ'}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Device Controls */}
          <div>
            <Card className="bg-gray-800 border-gray-700">
              <CardHeader>
                <CardTitle className="text-white flex items-center">
                  <Settings className="w-5 h-5 mr-2" />
                  Điều khiển thiết bị
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Auto Feeding Toggle */}
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Cho ăn tự động</span>
                  <button
                    onClick={handleAutoFeedingToggle}
                    className={`w-12 h-6 rounded-full relative transition-all duration-300 ease-in-out transform hover:scale-105 active:scale-95 ${
                      autoFeeding ? "bg-blue-600 hover:bg-blue-500" : "bg-gray-600 hover:bg-gray-500"
                    }`}
                  >
                    <div
                      className={`w-5 h-5 bg-white rounded-full absolute top-0.5 transition-all duration-300 ease-in-out shadow-lg ${
                        autoFeeding ? "right-0.5 transform" : "left-0.5"
                      }`}
                    />
                  </button>
                </div>

                {/* Area Sensor Toggle */}
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Cảm biến giới hạn khu vực</span>
                  <button
                    onClick={handleAreaSensorToggle}
                    className={`w-12 h-6 rounded-full relative transition-all duration-300 ease-in-out transform hover:scale-105 active:scale-95 ${
                      areaSensor ? "bg-blue-600 hover:bg-blue-500" : "bg-gray-600 hover:bg-gray-500"
                    }`}
                  >
                    <div
                      className={`w-5 h-5 bg-white rounded-full absolute top-0.5 transition-all duration-300 ease-in-out shadow-lg ${
                        areaSensor ? "right-0.5" : "left-0.5"
                      }`}
                    />
                  </button>
                </div>

                {/* Laser Game Toggle */}
                <div className="flex items-center justify-between">
                  <span className="text-gray-300">Trò chơi laser</span>
                  <button
                    onClick={handleLaserGameToggle}
                    className={`w-12 h-6 rounded-full relative transition-all duration-300 ease-in-out transform hover:scale-105 active:scale-95 ${
                      laserGame ? "bg-blue-600 hover:bg-blue-500" : "bg-gray-600 hover:bg-gray-500"
                    }`}
                  >
                    <div
                      className={`w-5 h-5 bg-white rounded-full absolute top-0.5 transition-all duration-300 ease-in-out shadow-lg ${
                        laserGame ? "right-0.5" : "left-0.5"
                      }`}
                    />
                  </button>
                </div>

                <div className="mt-6">
                  <div className="text-gray-300 mb-2">Điều khiển máy lạnh</div>
                  <button
                    onClick={() => handleTemperatureChange(1)}
                    disabled={isLoading}
                    className="w-full bg-pink-500 hover:bg-pink-400 active:bg-pink-600 disabled:bg-gray-500 disabled:cursor-not-allowed transition-all duration-200 transform hover:scale-[1.02] active:scale-[0.98] disabled:transform-none rounded-lg p-4 mb-2"
                  >
                    <div className="flex items-center justify-center gap-2">
                      <Plus className="w-4 h-4 text-white" />
                      <span className="text-white font-bold">
                        {isLoading ? "ĐANG XỬ LÝ..." : "TĂNG NHIỆT ĐỘ"}
                      </span>
                    </div>
                  </button>
                  <button
                    onClick={() => handleTemperatureChange(-1)}
                    disabled={isLoading}
                    className="w-full bg-pink-500 hover:bg-pink-400 active:bg-pink-600 disabled:bg-gray-500 disabled:cursor-not-allowed transition-all duration-200 transform hover:scale-[1.02] active:scale-[0.98] disabled:transform-none rounded-lg p-4"
                  >
                    <div className="flex items-center justify-center gap-2">
                      <Minus className="w-4 h-4 text-white" />
                      <span className="text-white font-bold">
                        {isLoading ? "ĐANG XỬ LÝ..." : "GIẢM NHIỆT ĐỘ"}
                      </span>
                    </div>
                  </button>
                </div>

                <div className="mt-6 space-y-2 text-sm text-gray-400">
                  <div>
                    Trạng thái kết nối: <span className="text-green-400">Đã kết nối</span>
                  </div>
                  <div>
                    Thiết bị hoạt động: <span className="text-white">8/10</span>
                  </div>
                  <div>
                    Mèo trong khu vực: <span className={areaSensor ? "text-green-400" : "text-red-400"}>
                      {areaSensor ? "An toàn" : "Không phát hiện"}
                    </span>
                  </div>
                  <div>
                    Lần cho ăn cuối: <span className="text-white">{formatLastFeedTime(lastFeedTime)}</span>
                  </div>
                  <div className="border-t border-gray-600 pt-2 mt-2">
                    🚨 Hệ thống cảnh báo: <span className="text-green-400 font-medium">Hoạt động</span>
                  </div>
                  <div className="text-xs text-gray-500">
                    📧 Email + 📱 Mobile push khi thức ăn/nước &lt; 20%
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>

        {/* Bottom Row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
          {/* Food and Water Status */}
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center">
                <Utensils className="w-5 h-5 mr-2" />
                Trạng thái thức ăn và nước trong bình
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-gray-300">Thức ăn khô</span>
                  <span className={`${
                    foodPercentage < 20 ? 'text-red-400 animate-pulse' : 
                    foodPercentage < 50 ? 'text-yellow-400' : 'text-green-400'
                  }`}>
                    {Math.round(foodPercentage)}%
                    {foodPercentage < 20 && <span className="ml-1">⚠️</span>}
                  </span>
                </div>
                <Progress 
                  value={foodPercentage} 
                  className={`h-2 bg-gray-700 ${foodPercentage < 20 ? 'animate-pulse' : ''}`}
                />
                <div className="flex justify-between text-xs text-gray-400 mt-1">
                  <span>Lần cuối: {formatLastFeedTime(lastFeedTime)}</span>
                  <span>{autoFeeding ? 'Tự động bật' : 'Chế độ thủ công'}</span>
                </div>
                {foodPercentage < 20 && (
                  <div className="text-xs text-red-400 mt-1 flex items-center">
                    ⚠️ Thức ăn sắp hết! {autoFeeding ? 'Sẽ tự động cho ăn.' : 'Vui lòng cho ăn thủ công.'}
                  </div>
                )}
                <Button
                  className="w-full mt-3 bg-purple-600 hover:bg-purple-500 active:bg-purple-700 disabled:bg-gray-500 disabled:cursor-not-allowed transition-all duration-200 transform hover:scale-[1.02] active:scale-[0.98] disabled:transform-none shadow-lg hover:shadow-xl"
                  onClick={handleManualFeed}
                  disabled={isLoading}
                >
                  {isLoading ? "⏳ Đang cho ăn..." : "▶ Cho thức ăn khô ngay"}
                </Button>
              </div>

              <div>
                <div className="flex justify-between items-center mb-2">
                  <span className="text-gray-300">Nước uống</span>
                  <span className={`${
                    waterPercentage < 20 ? 'text-red-400 animate-pulse' : 
                    waterPercentage < 50 ? 'text-yellow-400' : 'text-green-400'
                  }`}>
                    {Math.round(waterPercentage)}%
                    {waterPercentage < 20 && <span className="ml-1">⚠️</span>}
                  </span>
                </div>
                <Progress 
                  value={waterPercentage} 
                  className={`h-2 bg-gray-700 ${waterPercentage < 20 ? 'animate-pulse' : ''}`}
                />
                <div className="flex justify-between text-xs text-gray-400 mt-1">
                  <span>Trạng thái: {waterPercentage > 80 ? 'Đầy' : waterPercentage > 50 ? 'Đủ dùng' : 'Cần đổ thêm'}</span>
                  <span>Luôn sẵn sàng</span>
                </div>
                {waterPercentage < 20 && (
                  <div className="text-xs text-red-400 mt-1 flex items-center">
                    ⚠️ Nước sắp hết! Vui lòng đổ thêm nước.
                  </div>
                )}
                <Button
                  className="w-full mt-3 bg-blue-600 hover:bg-blue-500 active:bg-blue-700 disabled:bg-gray-500 disabled:cursor-not-allowed transition-all duration-200 transform hover:scale-[1.02] active:scale-[0.98] disabled:transform-none shadow-lg hover:shadow-xl"
                  onClick={handleRefillWater}
                  disabled={isLoading}
                >
                  {isLoading ? "⏳ Đang đổ nước..." : "▶ Đổ nước ngay"}
                </Button>
              </div>

              <div className="text-xs text-gray-400 space-y-1">
                <div>⏰ Lịch cho ăn: 7:00, 12:00, 18:00</div>
                <div>Số lần cho mèo chơi hôm nay: 7</div>
                <div>
                  Cảnh báo sắp hết: <span className="text-orange-400">Dưới 20%</span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Activity Chart */}
          <Card className="bg-gray-800 border-gray-700">
            <CardHeader>
              <CardTitle className="text-white flex items-center">
                <Activity className="w-5 h-5 mr-2" />
                Biểu đồ hoạt động
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="mb-4">
                <h4 className="text-gray-300 mb-2">Biểu đồ thời gian</h4>
                <ChartContainer
                  config={{
                    humidity: {
                      label: "Độ ẩm",
                      color: "#3b82f6", // Blue
                    },
                    temperature: {
                      label: "Nhiệt độ", 
                      color: "#f97316", // Orange
                    },
                  }}
                  className="h-[128px] w-full"
                >
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={historicalData.length > 0 ? 
                        historicalData.slice(-24).map((entry, index) => ({
                          name: new Date(entry.timestamp).toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' }),
                          humidity: Math.round(entry.humidity),
                          temperature: Math.round(entry.temperature)
                        })) :
                        [
                          { name: "00:00", humidity: 40, temperature: 22 },
                          { name: "04:00", humidity: 45, temperature: 25 },
                          { name: "08:00", humidity: 38, temperature: 28 },
                          { name: "12:00", humidity: 50, temperature: 26 },
                          { name: "16:00", humidity: 42, temperature: 23 },
                          { name: "20:00", humidity: 48, temperature: 20 },
                          { name: "24:00", humidity: 43, temperature: 21 },
                        ]
                      }
                      margin={{ top: 5, right: 10, left: 10, bottom: 0 }}
                    >
                      <CartesianGrid vertical={false} strokeDasharray="3 3" stroke="#374151" />
                      <XAxis
                        dataKey="name"
                        tickLine={false}
                        axisLine={false}
                        tickMargin={8}
                        tickFormatter={(value) => value.split(":")[0]}
                        className="text-xs text-gray-400"
                      />
                      <YAxis tickLine={false} axisLine={false} tickMargin={8} className="text-xs text-gray-400" />
                      <ChartTooltip cursor={false} content={<ChartTooltipContent />} />
                      <Line
                        type="monotone"
                        dataKey="humidity"
                        stroke="#3b82f6"
                        strokeWidth={2}
                        dot={false}
                      />
                      <Line
                        type="monotone"
                        dataKey="temperature"
                        stroke="#f97316"
                        strokeWidth={2}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </ChartContainer>
                <div className="flex justify-between text-xs text-gray-400 mt-2">
                  <span className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-blue-500"></span> Độ ẩm
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-2 h-2 rounded-full bg-orange-500"></span> Nhiệt độ
                  </span>
                </div>
              </div>

              <div>
                <h4 className="text-gray-300 mb-2">Biểu đồ theo đối cảnh báo</h4>
                <div className="relative w-24 h-24 mx-auto">
                  <svg className="w-24 h-24 transform -rotate-90" viewBox="0 0 100 100">
                    <circle cx="50" cy="50" r="40" stroke="rgb(75, 85, 99)" strokeWidth="8" fill="none" />
                    <circle
                      cx="50"
                      cy="50"
                      r="40"
                      stroke="rgb(34, 197, 94)"
                      strokeWidth="8"
                      fill="none"
                      strokeDasharray={`${(70 / 100) * 251.2} 251.2`}
                      strokeLinecap="round"
                    />
                    <circle
                      cx="50"
                      cy="50"
                      r="30"
                      stroke="rgb(168, 85, 247)"
                      strokeWidth="6"
                      fill="none"
                      strokeDasharray={`${(20 / 100) * 188.4} 188.4`}
                      strokeLinecap="round"
                    />
                    <circle
                      cx="50"
                      cy="50"
                      r="20"
                      stroke="rgb(236, 72, 153)"
                      strokeWidth="4"
                      fill="none"
                      strokeDasharray={`${(10 / 100) * 125.6} 125.6`}
                      strokeLinecap="round"
                    />
                  </svg>
                </div>
                <div className="text-xs text-gray-400 space-y-1 mt-2">
                  <div>● Số lần nhận nút chơi</div>
                  <div>● Số lần ăn</div>
                  <div>● Số lần uống nước</div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
